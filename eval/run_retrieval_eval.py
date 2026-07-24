"""Retrieval accuracy evaluation — the RAG-side counterpart to
run_extraction_eval.py. Runs the exact hybrid-search + rerank pipeline
rag.answer() uses (see rag.py) against hand-labelled queries and checks
whether the right passage lands in the top-k retrieved chunks.

Queries are deliberately paraphrased away from the source wording (see
rag_eval_queries.json's _README) so this fails loudly if retrieval quality
regresses to lexical-only matching — e.g. if the semantic backend silently
falls back to TF-IDF (see index.py's build_index), or a chunking change
buries the target passage in an oversized, unranked chunk.

Runs fully offline against the bundled synthetic sample corpus — no
internet or API keys required, same as `pytest`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.chunker import chunk_corpus
from finsight.config import SAMPLE_DIR, settings
from finsight.ingest import load_corpus
import finsight.index as index_mod
from finsight.rag import _rerank

# Isolate this run's Chroma collection from the one a live `streamlit run
# app.py` process persists to data/index — both use the same hardcoded
# "filings" collection name, and running this eval while the app is mid
# index-build would delete/replace its in-progress collection (see
# tests/conftest.py for the same fix applied to the test suite).
index_mod.INDEX_DIR = Path(tempfile.mkdtemp(prefix="finsight-eval-index-"))

TOP_K = 5
PASS_THRESHOLD = 0.85


def main() -> int:
    payload = json.loads((ROOT / "eval" / "rag_eval_queries.json").read_text())
    queries = payload["queries"]

    docs = load_corpus(SAMPLE_DIR)
    index, backend = index_mod.build_index(chunk_corpus(docs))
    print(f"Retrieval evaluation — {len(queries)} labelled queries, backend={backend}\n" + "-" * 60)

    hits = 0
    for q in queries:
        pool = index.search(q["question"], max(settings.rerank_pool, TOP_K * 3),
                            ticker=q.get("ticker"))
        top = _rerank(q["question"], pool)[:TOP_K]
        text = " ".join(c.text for c, _ in top).lower()
        ok = any(needle.lower() in text for needle in q["expect_any"])
        hits += ok
        print(f"[{'OK' if ok else 'MISS'}] {q['question']}")
        if not ok:
            print(f"       expected one of {q['expect_any']!r} in top-{TOP_K}, got: "
                  f"{[c.citation for c, _ in top]}")

    acc = hits / len(queries)
    print("-" * 60)
    print(f"OVERALL {hits}/{len(queries)} ({acc:.0%})  "
          f"{'PASS' if acc >= PASS_THRESHOLD else 'BELOW TARGET'} (>= {PASS_THRESHOLD:.0%})")
    return 0 if acc >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
