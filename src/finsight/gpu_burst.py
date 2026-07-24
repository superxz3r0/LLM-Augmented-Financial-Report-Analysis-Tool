"""Generic ephemeral Verda GPU runner: JSON payload in, JSON result out.

Used only by scripts/precompute.py (the Streamlit app never touches
this module — it just reads data/derived/). Flow:

    1. credentials       env vars first, then AWS SSM Parameter Store
    2. create instance   Verda public API (A100 by default)
    3. wait              status == "running", then sshd reachable
    4. upload            worker script + payload.json (scp)
    5. run               pip install + the worker (ssh)
    6. download          result.json (scp)
    7. delete instance   in a ``finally`` — a crash never leaves a GPU billing

One-time setup
--------------
* upload your SSH *public* key to the Verda account (console or API)
* store secrets in env vars or SSM under ``$FINSIGHT_SSM_PREFIX``
  (default ``/finsight/prod``): ``verda-client-id``,
  ``verda-client-secret``, ``ssh-private-key``
* ``pip install verda boto3`` on the machine running precompute
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[2]      # src/finsight/ -> project root
_LOCK_PATH = _ROOT / "data" / "burst.lock"
WORKER = _ROOT / "scripts" / "score_sentiment_gpu.py"

SSM_PREFIX = os.environ.get("FINSIGHT_SSM_PREFIX", "/finsight/prod")
INSTANCE_TYPE = os.environ.get("VERDA_INSTANCE_TYPE", "1A100.22V")
IMAGE = os.environ.get("VERDA_IMAGE", "ubuntu-24.04-cuda-12.8-open-docker")
LOCATION = os.environ.get("VERDA_LOCATION", "FIN-01")
SSH_USER = os.environ.get("VERDA_SSH_USER", "root")   # Verda images default to root

_SSH_OPTS = ["-C",                               # compress: payloads are big JSON
             "-o", "StrictHostKeyChecking=no",   # ephemeral host, no pinned key
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "ConnectTimeout=10",
             "-o", "LogLevel=ERROR"]

# Fresh Ubuntu 24.04 is PEP 668 "externally managed" -> --break-system-packages
_BOOTSTRAP = ("command -v pip3 >/dev/null 2>&1 || "
              "(apt-get -qq update && apt-get -qq -y install python3-pip); "
              "python3 -m pip install -q --break-system-packages torch transformers")


# ------------------------------------------------------------------ secrets

def _ssm_get(name: str) -> str | None:
    """Fetch a SecureString from SSM; None if unavailable (no boto3, no
    permission, parameter missing). Deliberately uncached — secrets stay
    out of long-lived process caches; bursts are rare."""
    try:
        import boto3
        return boto3.client("ssm").get_parameter(
            Name=f"{SSM_PREFIX}/{name}",
            WithDecryption=True)["Parameter"]["Value"]
    except Exception:
        return None


def _secret(env: str, ssm_name: str) -> str | None:
    return os.environ.get(env) or _ssm_get(ssm_name)


@functools.lru_cache(maxsize=1)
def verda_credentials_available() -> bool:
    return bool(_secret("VERDA_CLIENT_ID", "verda-client-id")
                and _secret("VERDA_CLIENT_SECRET", "verda-client-secret"))


def _verda_client():
    from verda import VerdaClient
    cid = _secret("VERDA_CLIENT_ID", "verda-client-id")
    csec = _secret("VERDA_CLIENT_SECRET", "verda-client-secret")
    if not (cid and csec):
        raise RuntimeError("Verda credentials not found in env or SSM")
    return VerdaClient(cid, csec)


class _TempKey:
    """SSH private key from env/SSM -> chmod-600 temp file, deleted on exit."""

    def __enter__(self) -> str:
        pem = _secret("VERDA_SSH_PRIVATE_KEY", "ssh-private-key")
        if not pem:
            raise RuntimeError("SSH private key not found in env or SSM")
        fd, self.path = tempfile.mkstemp(prefix="verda_key_")
        with os.fdopen(fd, "w") as f:
            f.write(pem if pem.endswith("\n") else pem + "\n")
        os.chmod(self.path, 0o600)
        return self.path

    def __exit__(self, *exc) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass


# -------------------------------------------------------------------- lock

def _acquire_lock() -> None:
    """Best-effort guard: two precompute runs must not launch two GPUs."""
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_PATH.exists():
        if time.time() - _LOCK_PATH.stat().st_mtime < 1800:
            raise RuntimeError(
                "another GPU burst appears to be running — wait for it, or "
                f"remove {_LOCK_PATH} if it is stale")
        _LOCK_PATH.unlink()                          # stale (>30 min)
    _LOCK_PATH.write_text(str(os.getpid()))


def _release_lock() -> None:
    _LOCK_PATH.unlink(missing_ok=True)


# ------------------------------------------------------------- ssh helpers

def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ssh(ip: str, key: str, remote_cmd: str, timeout: int = 1800) -> str:
    p = _run(["ssh", "-i", key, *_SSH_OPTS, f"{SSH_USER}@{ip}", remote_cmd],
             timeout)
    if p.returncode != 0:
        raise RuntimeError(f"ssh failed ({remote_cmd[:60]}…): {p.stderr[-800:]}")
    return p.stdout


def _scp(key: str, src: str, dst: str, timeout: int = 600) -> None:
    p = _run(["scp", "-i", key, *_SSH_OPTS, src, dst], timeout)
    if p.returncode != 0:
        raise RuntimeError(f"scp failed: {p.stderr[-800:]}")


def _wait_for_ssh(ip: str, key: str, timeout: int = 180) -> None:
    """`running` does not mean sshd is up yet — retry until it answers."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _ssh(ip, key, "true", timeout=20)
            return
        except Exception:
            time.sleep(8)
    raise TimeoutError("sshd did not come up in time")


# ------------------------------------------------------------ provisioning

def _wait_until_running(client, instance_id, timeout: int = 600) -> str:
    """Poll the API until the instance is `running`; return its public IP.

    Method/attribute names follow the official SDK (instances.get_by_id,
    .status, .ip) — verify against the SDK docs if your version differs.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        inst = client.instances.get_by_id(instance_id)
        status = (getattr(inst, "status", "") or "").lower()
        ip = getattr(inst, "ip", None)
        if status == "running" and ip:
            return ip
        if status in ("error", "discontinued"):
            raise RuntimeError(f"instance entered state {status!r}")
        time.sleep(10)
    raise TimeoutError(f"instance not `running` within {timeout}s — "
                       "check the Verda console")


# ------------------------------------------------------------- main burst

def burst_score(payload: dict, log: Callable[[str], None] = print) -> dict:
    """Run the worker against `payload` on an ephemeral Verda GPU.

    `payload` is any JSON-serialisable object the worker understands
    (see scripts/score_sentiment_gpu.py). Returns the worker's JSON
    output parsed into a dict. The instance is deleted in a
    finally-block, so the worst case on a crash is one pre-paid
    10-minute billing unit.
    """
    from verda.constants import Actions

    if not payload:
        return {}
    _acquire_lock()
    client = _verda_client()
    inst = None
    try:
        key_ids = [k.id for k in client.ssh_keys.get()]
        if not key_ids:
            raise RuntimeError("no SSH public key registered on the Verda "
                               "account — upload one in the console first")

        log(f"Deploying {INSTANCE_TYPE} ({IMAGE}) in {LOCATION}…")
        inst = client.instances.create(
            instance_type=INSTANCE_TYPE, image=IMAGE, ssh_key_ids=key_ids,
            location_code=LOCATION, hostname="finsight-burst",
            description="FinSight precompute burst (auto-deleted)")

        ip = _wait_until_running(client, inst.id)
        log(f"Instance {inst.id} running at {ip}; waiting for SSH…")

        with _TempKey() as key, tempfile.TemporaryDirectory() as tmp:
            _wait_for_ssh(ip, key)

            payload_path = Path(tmp) / "payload.json"
            payload_path.write_text(json.dumps(payload))

            log("Uploading worker script and payload…")
            _scp(key, str(WORKER), f"{SSH_USER}@{ip}:worker.py")
            _scp(key, str(payload_path), f"{SSH_USER}@{ip}:payload.json")

            log("Installing torch + transformers (a few minutes on a fresh "
                "image — bake a golden OS volume to skip this)…")
            _ssh(ip, key, _BOOTSTRAP, timeout=1500)

            log("Running worker on the GPU…")
            _ssh(ip, key, "python3 worker.py payload.json result.json",
                 timeout=3600)

            result_path = Path(tmp) / "result.json"
            _scp(key, f"{SSH_USER}@{ip}:result.json", str(result_path))
            result = json.loads(result_path.read_text())
        log(f"Got {len(result)} results — deleting instance.")
        return result
    finally:
        if inst is not None:
            try:
                client.instances.action(inst.id, Actions.DELETE)
            except Exception as e:                    # noqa: BLE001
                log(f"WARNING: could not delete instance {inst.id} ({e}) — "
                    "delete it manually in the Verda console to stop billing!")
        _release_lock()
