from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from .config import PROJECT_ROOT

_ENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_OPENAI_KEY_PREFIXES = ("sk-", "sk-proj-")
_RETRYABLE_CATEGORIES = {"rate_limit", "timeout", "network_error"}
T = TypeVar("T")


@dataclass(frozen=True)
class ApiKeyStatus:
    provider: str
    env_var: str
    configured: bool
    valid: bool
    masked: str
    length: int
    message: str


class LLMCallError(RuntimeError):
    def __init__(self, provider: str, category: str, message: str):
        super().__init__(message)
        self.provider = provider
        self.category = category


def mask_api_key(key: str | None) -> str:
    cleaned = _clean_api_key(key)
    if not cleaned:
        return "<not set>"
    if len(cleaned) < 8:
        return "<invalid or too short>"
    return f"{cleaned[:3]}...{cleaned[-2:]}"


def load_environment(env_path: Path | None = None) -> None:
    """Load .env values without requiring python-dotenv at runtime."""
    path = env_path or PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(path if path.exists() else None, override=False)
        return

    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("export "):
            line = line.removeprefix("export ").strip()
        match = _ENV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), _clean_api_key(match.group(2))
        if key and value and key not in os.environ:
            os.environ[key] = value


def api_key_status(
    provider: str,
    env_var: str,
    aliases: tuple[str, ...] = (),
    *,
    env_path: Path | None = None,
) -> ApiKeyStatus:
    load_environment(env_path)
    raw = os.getenv(env_var)
    if not raw:
        for alias in aliases:
            raw = os.getenv(alias)
            if raw:
                os.environ[env_var] = _clean_api_key(raw)
                break

    cleaned = _clean_api_key(raw)
    if raw and cleaned != raw:
        os.environ[env_var] = cleaned

    length = len(cleaned)
    if not cleaned:
        return ApiKeyStatus(
            provider, env_var, False, False, "<not set>", 0,
            f"{env_var} is not configured.",
        )
    if provider.lower() == "openai":
        valid = length >= 20 and cleaned.startswith(_OPENAI_KEY_PREFIXES)
        detail = (
            f"{env_var} is set but does not look like an OpenAI API key "
            f"(length={length}, masked={mask_api_key(cleaned)})."
        )
    else:
        valid = length >= 8
        detail = (
            f"{env_var} is set but is too short "
            f"(length={length}, masked={mask_api_key(cleaned)})."
        )
    return ApiKeyStatus(
        provider, env_var, True, valid, mask_api_key(cleaned), length,
        f"{env_var} is configured." if valid else detail,
    )


def openai_error(exc: Exception) -> LLMCallError:
    name = type(exc).__name__
    text = str(exc)
    lowered = f"{name} {text}".lower()
    if isinstance(exc, ModuleNotFoundError):
        category = "configuration_failure"
        message = "The openai package is not installed. Run pip install -r requirements.txt."
    elif "authentication" in lowered or "api key" in lowered or "401" in lowered:
        category = "authentication_error"
        message = "OpenAI authentication failed. Check OPENAI_API_KEY."
    elif "notfound" in lowered or "model" in lowered and "not found" in lowered or "404" in lowered:
        category = "model_not_found"
        message = "OpenAI model was not found or is unavailable for this account."
    elif "ratelimit" in lowered or "rate limit" in lowered or "429" in lowered:
        category = "rate_limit"
        message = "OpenAI rate limit reached."
    elif "timeout" in lowered or "timed out" in lowered:
        category = "timeout"
        message = "OpenAI request timed out."
    elif "connection" in lowered or "network" in lowered:
        category = "network_error"
        message = "OpenAI network error."
    elif "badrequest" in lowered or "invalid request" in lowered or "400" in lowered:
        category = "invalid_request"
        message = "OpenAI rejected the request."
    else:
        category = "generation_failure"
        message = f"OpenAI generation failed ({name})."
    return LLMCallError("openai", category, message)


def gemini_error(exc: Exception) -> LLMCallError:
    if isinstance(exc, ModuleNotFoundError):
        return LLMCallError(
            "gemini", "configuration_failure",
            "The google-generativeai package is not installed. Run pip install -r requirements.txt.",
        )
    return LLMCallError("gemini", "generation_failure", f"Gemini generation failed ({type(exc).__name__}).")


def call_with_retries(fn: Callable[[], T], classifier: Callable[[Exception], LLMCallError],
                      attempts: int = 3, base_delay: float = 0.5) -> T:
    last_error: LLMCallError | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            err = classifier(exc)
            last_error = err
            if err.category not in _RETRYABLE_CATEGORIES or attempt == attempts - 1:
                raise err from exc
            time.sleep(base_delay * (2 ** attempt))
    assert last_error is not None
    raise last_error


def _clean_api_key(key: str | None) -> str:
    if key is None:
        return ""
    return key.strip().strip("\"'").strip()
