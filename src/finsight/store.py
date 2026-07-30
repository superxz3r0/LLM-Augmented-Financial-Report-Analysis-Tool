from __future__ import annotations

import json
import sqlite3

from .config import DATA_DIR

EXTRACTOR_VERSION = 3
_DB = DATA_DIR / "finsight.db"


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        doc_id TEXT, version INTEGER, payload TEXT,
        PRIMARY KEY (doc_id, version))""")
    return con


def get_signals(doc_id: str) -> dict | None:
    con = _conn()
    try:
        with con:
            row = con.execute("SELECT payload FROM signals WHERE doc_id=? AND version=?",
                              (doc_id, EXTRACTOR_VERSION)).fetchone()
    finally:
        con.close()
    return json.loads(row[0]) if row else None


def put_signals(doc_id: str, payload: dict) -> None:
    con = _conn()
    try:
        with con:
            con.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?)",
                        (doc_id, EXTRACTOR_VERSION, json.dumps(payload)))
    finally:
        con.close()


def cached_extract(doc, use_llm: bool = True) -> dict:
    cached = get_signals(doc.doc_id)
    if cached is not None:
        return cached
    from .extract import extract_signals
    payload = extract_signals(doc, use_llm=use_llm).as_dict()
    put_signals(doc.doc_id, payload)
    return payload
