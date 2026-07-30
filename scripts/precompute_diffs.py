"""Precompute filing diffs on a GPU box and cache them to JSON.

The Filing-diff tab runs a semantic (embedding-based) diff on demand, which
is slow on CPU-only machines. This script computes every diff the UI can ask
for — each filing against the *next filing of the same form* (10-K vs next
10-K, 10-Q vs next 10-Q) — once, on whatever hardware runs it (ideally the
GPU box), and writes the results to data/diff_cache.json.

Copy that JSON back to your local data/ dir and the app will read diffs from
it instantly instead of recomputing (see app.py's Filing diff tab).

Run from the project root:
    FINSIGHT_DATA_DIR=$(pwd)/data PYTHONPATH=src python scripts/precompute_diffs.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.config import FILINGS_DIR, SAMPLE_DIR, DATA_DIR
from finsight.ingest import load_corpus
import finsight.diff as diff_mod

# Same key scheme the app uses to look a diff up: old_doc_id + "|" + new_doc_id.
CACHE_PATH = DATA_DIR / "diff_cache.json"


def consecutive_same_form_pairs(docs):
    """Yield (old, new) pairs: each filing with the next filing of the same
    form for the same ticker. Mirrors the app's Filing-diff pairing rule."""
    by_ticker = {}
    for d in docs:
        by_ticker.setdefault(d.ticker, []).append(d)

    for ticker, ds in by_ticker.items():
        for form in {d.form for d in ds}:
            same_form = sorted([d for d in ds if d.form == form],
                               key=lambda d: d.date)
            for older, newer in zip(same_form, same_form[1:]):
                yield older, newer


def main() -> int:
    docs = load_corpus(FILINGS_DIR, SAMPLE_DIR)
    pairs = list(consecutive_same_form_pairs(docs))
    print(f"Found {len(pairs)} same-form consecutive filing pairs to diff.")

    cache: dict[str, list[dict]] = {}
    for n, (old, new) in enumerate(pairs, 1):
        key = f"{old.doc_id}|{new.doc_id}"
        print(f"[{n}/{len(pairs)}] {key} …", flush=True)
        items = diff_mod.diff_documents(old, new)
        cache[key] = [asdict(i) for i in items]

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(cache)} cached diffs to {CACHE_PATH}")
    print("Copy this file back to your local data/ directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
