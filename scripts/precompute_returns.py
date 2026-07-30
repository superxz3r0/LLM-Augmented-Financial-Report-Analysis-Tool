"""Precompute signal→return regressions and cache them to JSON.

The Signal→returns tab runs two studies per company — filing sentiment and
disclosure-change (substantive edits vs. the prior same-form filing) — each a
grouped OLS on 5- and 20-day forward returns. That requires per-filing signal
extraction (FinBERT) and forward-price fetches, which are slow/flaky on a
laptop. This script runs every per-company study once, ideally on the GPU box
with network access, and writes results to data/returns_cache.json.

Copy that JSON back to your local data/ dir and the app reads regressions from
it instantly instead of recomputing (see app.py's Signal→returns tab).

The disclosure-change signal reuses the diff cache (data/diff_cache.json) when
present, so run scripts/precompute_diffs.py first for best speed.

Run from the project root:
    FINSIGHT_DATA_DIR=$(pwd)/data PYTHONPATH=src python scripts/precompute_returns.py
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
from finsight.store import cached_extract
from finsight import returns as ret_mod
from finsight import diff as diff_mod

CACHE_PATH = DATA_DIR / "returns_cache.json"
DIFF_CACHE_PATH = DATA_DIR / "diff_cache.json"


def _load_diff_cache() -> dict:
    if not DIFF_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(DIFF_CACHE_PATH.read_text())
    except Exception:
        return {}


_DIFF_CACHE = _load_diff_cache()


def _clean_controls(controls: dict) -> dict:
    out = {}
    for name, val in (controls or {}).items():
        try:
            out[name] = [float(x) for x in val]
        except TypeError:
            out[name] = float(val)
    return out


def _serialise(results):
    out = []
    for r in results:
        d = asdict(r)
        d["controls"] = _clean_controls(d.get("controls", {}))
        out.append(d)
    return out


def _serialise_study(grouped):
    return {label: _serialise(results) for label, results in grouped.items()}


def _group_counts(rows):
    counts = {}
    for r in rows:
        g = ret_mod.doc_group(r.get("form", ""))
        counts[g] = counts.get(g, 0) + 1
    return counts


def _disclosure_change(prev, doc):
    key = f"{prev.doc_id}|{doc.doc_id}"
    cached = _DIFF_CACHE.get(key)
    if cached is not None:
        return sum(1 for it in cached
                   if it.get("kind") in ("new", "substantive", "removed"))
    items = diff_mod.diff_documents(prev, doc)
    return sum(1 for it in items
               if it.kind in ("new", "substantive", "removed"))


def _rows_for(company_docs):
    docs_sorted = sorted(company_docs, key=lambda d: d.date)
    prev_same_form = {}
    seen = {}
    for d in docs_sorted:
        if d.form != "TRANSCRIPT" and d.form in seen:
            prev_same_form[d.doc_id] = seen[d.form]
        seen[d.form] = d

    rows = []
    for d in docs_sorted:
        s = cached_extract(d, use_llm=False)
        prev = prev_same_form.get(d.doc_id)
        dc = _disclosure_change(prev, d) if prev is not None else None
        rows.append({"ticker": d.ticker, "date": d.date, "form": d.form,
                     "signal": s["sentiment_score"], "disclosure_change": dc})
    return rows


def main() -> int:
    docs = load_corpus(FILINGS_DIR, SAMPLE_DIR)
    real_docs = [d for d in docs if d.path.parent.name != "sample"]
    if not real_docs:
        print("No real filings found — nothing to precompute.")
        return 1

    tickers = sorted({d.ticker for d in real_docs})
    cache = {}

    for n, t in enumerate(tickers, 1):
        print(f"[{n}/{len(tickers)}] {t} …", flush=True)
        rows = _rows_for([d for d in real_docs if d.ticker == t])

        sentiment = ret_mod.run_study_grouped(rows, signal_field="signal")
        dc_rows = [r for r in rows if r.get("disclosure_change") is not None]
        disclosure = (ret_mod.run_study_grouped(dc_rows,
                                                signal_field="disclosure_change")
                      if len(dc_rows) >= 5 else {})

        cache[t] = {
            "sentiment": _serialise_study(sentiment),
            "disclosure": _serialise_study(disclosure),
            "group_counts": _group_counts(rows),
            "dc_group_counts": _group_counts(dc_rows),
        }

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\nWrote studies for {len(cache)} companies to {CACHE_PATH}")
    print("Copy this file back to your local data/ directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
