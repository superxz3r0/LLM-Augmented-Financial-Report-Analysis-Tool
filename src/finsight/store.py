from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from typing import Callable, Iterable

from .config import DATA_DIR, settings
from .sentiment import DEFAULT_MAX_SENTENCES, SentimentResult

EXTRACTOR_VERSION = 2
REGRESSION_SENTIMENT_VERSION = 1
REGRESSION_TEXT_CHARS = 20_000
REGRESSION_MAX_SENTENCES = DEFAULT_MAX_SENTENCES
_DB = DATA_DIR / "finsight.db"


@contextmanager
def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB, timeout=30)
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS signals (
            doc_id TEXT, version INTEGER, payload TEXT,
            PRIMARY KEY (doc_id, version))""")
        con.execute("""CREATE TABLE IF NOT EXISTS regression_sentiment (
            doc_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            model_key TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            score REAL NOT NULL,
            n_sentences INTEGER NOT NULL,
            backend TEXT NOT NULL,
            breakdown TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (doc_id, version, model_key))""")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def get_signals(doc_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT payload FROM signals WHERE doc_id=? AND version=?",
                          (doc_id, EXTRACTOR_VERSION)).fetchone()
    return json.loads(row[0]) if row else None


def put_signals(doc_id: str, payload: dict) -> None:
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?)",
                    (doc_id, EXTRACTOR_VERSION, json.dumps(payload)))


def cached_extract(doc, use_llm: bool = True) -> dict:
    cached = get_signals(doc.doc_id)
    if cached is not None:
        return cached
    from .extract import extract_signals
    payload = extract_signals(doc, use_llm=use_llm).as_dict()
    put_signals(doc.doc_id, payload)
    return payload


def regression_sentiment_model_key(
    *,
    max_chars: int = REGRESSION_TEXT_CHARS,
    max_sentences: int = REGRESSION_MAX_SENTENCES,
) -> str:
    """Identify every input/model choice that can change a cached score."""
    return (
        f"model={settings.finbert_model}|max_chars={max_chars}"
        f"|max_sentences={max_sentences}|sentence_split=v1"
        "|aggregation=mean_positive_minus_negative_v1"
    )


def regression_text_hash(text: str) -> str:
    """Hash the exact text prefix supplied to the regression scorer."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_regression_sentiments(
    text_hashes: dict[str, str],
    model_key: str,
) -> dict[str, SentimentResult]:
    """Return valid cached scores for the requested document hashes.

    One query loads the rows for this model configuration; matching the text
    hash in Python avoids SQLite's parameter limit for an overall-corpus run.
    """
    if not text_hashes:
        return {}
    with _conn() as con:
        rows = con.execute(
            """SELECT doc_id, text_hash, score, n_sentences, backend, breakdown
               FROM regression_sentiment
               WHERE version=? AND model_key=?""",
            (REGRESSION_SENTIMENT_VERSION, model_key),
        ).fetchall()

    cached: dict[str, SentimentResult] = {}
    for doc_id, text_hash, score, n_sentences, backend, breakdown in rows:
        if text_hashes.get(doc_id) != text_hash:
            continue
        try:
            detail = json.loads(breakdown)
        except (TypeError, json.JSONDecodeError):
            continue
        cached[doc_id] = SentimentResult(
            score=float(score),
            n_sentences=int(n_sentences),
            backend=backend,
            breakdown=detail,
        )
    return cached


def put_regression_sentiments(
    entries: Iterable[tuple[str, str, SentimentResult]],
    model_key: str,
) -> None:
    """Persist a group of newly computed results in one transaction."""
    values = [
        (
            doc_id,
            REGRESSION_SENTIMENT_VERSION,
            model_key,
            text_hash,
            result.score,
            result.n_sentences,
            result.backend,
            json.dumps(result.breakdown, sort_keys=True),
        )
        for doc_id, text_hash, result in entries
    ]
    if not values:
        return
    with _conn() as con:
        con.executemany(
            """INSERT INTO regression_sentiment
               (doc_id, version, model_key, text_hash, score, n_sentences,
                backend, breakdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id, version, model_key) DO UPDATE SET
                   text_hash=excluded.text_hash,
                   score=excluded.score,
                   n_sentences=excluded.n_sentences,
                   backend=excluded.backend,
                   breakdown=excluded.breakdown,
                   updated_at=CURRENT_TIMESTAMP""",
            values,
        )


def score_regression_sentiments_cached(
    items: list[tuple[str, str]],
    *,
    max_chars: int = REGRESSION_TEXT_CHARS,
    max_sentences: int = REGRESSION_MAX_SENTENCES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[SentimentResult], int, int]:
    """Return ordered scores, computing only cache misses in one batch call.

    ``items`` contains ``(doc_id, text)`` pairs. The exact truncated text is
    both hashed and scored, so the cache cannot validate one input while the
    model sees another. Duplicate document IDs are rejected because they
    would make an ordered cache lookup ambiguous.
    """
    if not items:
        return [], 0, 0
    doc_ids = [doc_id for doc_id, _text in items]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Regression sentiment requires unique document IDs")

    prepared = [(doc_id, text[:max_chars]) for doc_id, text in items]
    hashes = {doc_id: regression_text_hash(text) for doc_id, text in prepared}
    model_key = regression_sentiment_model_key(
        max_chars=max_chars,
        max_sentences=max_sentences,
    )
    cached = get_regression_sentiments(hashes, model_key)
    missing = [
        (idx, doc_id, text)
        for idx, (doc_id, text) in enumerate(prepared)
        if doc_id not in cached
    ]

    ordered: list[SentimentResult | None] = [cached.get(doc_id) for doc_id in doc_ids]
    if missing:
        from . import sentiment

        computed = sentiment.score_texts(
            [text for _idx, _doc_id, text in missing],
            max_sentences=max_sentences,
            progress=progress,
        )
        if len(computed) != len(missing):
            raise RuntimeError("Sentiment scorer returned an unexpected result count")

        cache_entries = []
        for (idx, doc_id, _text), result in zip(missing, computed):
            ordered[idx] = result
            # A lexicon fallback may reflect a temporary model failure. Do not
            # let it hide a recoverable FinBERT result on a later run.
            if result.backend != "lexicon":
                cache_entries.append((doc_id, hashes[doc_id], result))
        put_regression_sentiments(cache_entries, model_key)

    if any(result is None for result in ordered):
        raise RuntimeError("Missing sentiment result after cache merge")
    return [result for result in ordered if result is not None], len(cached), len(missing)
