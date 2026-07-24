"""Read/write layer for precomputed ("derived") artifacts.

Everything expensive — FinBERT inference, filing diffs, yfinance price
fetches — is computed by ``scripts/precompute.py`` and saved here as
plain JSON. The Streamlit app only *reads* these files, and falls back
to on-the-fly computation when one is missing, so the same codebase
runs unchanged on a laptop, on EC2, or anywhere else. The whole state
of the pipeline is one directory; syncing a deployment is one rsync:

    data/derived/
    ├── sentiment.json         {text_key: {score, n_sentences, backend,
    │                                      ticker, date, form}}
    ├── forward_returns.json   {generated_at, windows,
    │                           market:    {date: {"5": r, "20": r}},
    │                           by_ticker: {ticker: {date: {...}}}}
    └── diffs.json             {ticker: {"old_date|new_date":
                                         {old_form, new_form, items: […]}}}

``sentiment.json`` is keyed by sha1 of the *exact text scored*, so a
score is independent of ticker/mode/run and CPU- and GPU-produced
entries are interchangeable — provided both paths use the same sentence
split and model (see scripts/precompute.py, which reuses the app's own
splitting code for exactly this reason).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]      # src/finsight/ -> project root
DERIVED = _ROOT / "data" / "derived"
SENTIMENT_PATH = DERIVED / "sentiment.json"
RETURNS_PATH = DERIVED / "forward_returns.json"
DIFFS_PATH = DERIVED / "diffs.json"

# Signal definition for the returns-study sentiment score. Both the app's
# CPU fallback and precompute's defaults read these, so cache keys and
# scores always line up. None means "no limit" — Python's open slice:
# text[:None] is the whole text, sentences[:None] is every sentence.
SENT_CHARS = None            # was 20000 — now score the FULL filing text
SENT_MAX_SENTENCES = None    # was 200  — now score every sentence


def text_key(text: str) -> str:
    """Stable cache key — sha1 of exactly the text that gets scored."""
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)            # never leave a half-written artifact


# ------------------------------------------------------------- sentiment

def load_sentiment() -> dict:
    return _load(SENTIMENT_PATH)


def save_sentiment(entries: dict) -> None:
    _save(SENTIMENT_PATH, entries)


# ------------------------------------------------------- forward returns

def load_returns() -> dict:
    return _load(RETURNS_PATH)


def save_returns(obj: dict) -> None:
    _save(RETURNS_PATH, obj)


def returns_cover(art: dict, rows: list[dict]) -> bool:
    """True iff the precompute run saw every (ticker, date) in `rows`.

    An empty per-date dict still counts as covered — it means "checked;
    forward window not tradeable yet", which is exactly what a live
    yfinance run would conclude too.
    """
    by_t = art.get("by_ticker", {})
    mkt = art.get("market", {})
    for r in rows:
        dt = str(r["date"])
        if dt not in by_t.get(r["ticker"], {}) or dt not in mkt:
            return False
    return True


def run_study_offline(rows: list[dict], art: dict,
                      market_control: bool = True) -> list:
    """Same regressions as returns.run_study, but forward returns come
    from the artifact instead of yfinance. Reuses returns.ols directly
    so the maths cannot drift from the online path."""
    import numpy as np

    from .returns import ols

    windows = tuple(art.get("windows", (5, 20)))
    by_t, mkt = art.get("by_ticker", {}), art.get("market", {})

    enriched = []
    for r in rows:
        dt = str(r["date"])
        fr = {int(w): v for w, v in by_t.get(r["ticker"], {}).get(dt, {}).items()}
        mk = {int(w): v for w, v in mkt.get(dt, {}).items()}
        enriched.append({**r, "returns": fr, "market": mk})

    results = []
    for w in windows:
        sig = np.array([r["signal"] for r in enriched])
        ret = np.array([r["returns"].get(w, np.nan) for r in enriched])
        controls = None
        if market_control:
            controls = {"mkt": np.array([r["market"].get(w, np.nan)
                                         for r in enriched])}
        results.append(ols(sig, ret, w, controls))
    return results


# ----------------------------------------------------------------- diffs

def _pair_key(old_date: str, new_date: str) -> str:
    return f"{old_date}|{new_date}"


def load_diffs() -> dict:
    return _load(DIFFS_PATH)


def save_diffs(obj: dict) -> None:
    _save(DIFFS_PATH, obj)


def get_diff(art: dict, ticker: str, old_date: str, new_date: str):
    """Precomputed diff items as attribute-style objects (drop-in for
    diff.DiffItem in the UI), or None if this pair wasn't precomputed."""
    rec = art.get(ticker, {}).get(_pair_key(old_date, new_date))
    if rec is None:
        return None
    return [SimpleNamespace(**it) for it in rec["items"]]


# ------------------------------------------------------- status (sidebar)

def summary() -> dict:
    s, r, d = load_sentiment(), load_returns(), load_diffs()
    return {
        "sentiment": len(s),
        "returns_tickers": len(r.get("by_ticker", {})),
        "returns_generated": r.get("generated_at"),
        "diff_pairs": sum(len(v) for v in d.values()),
    }
