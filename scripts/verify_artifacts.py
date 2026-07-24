"""Verify that data/derived/ fully covers the current corpus.

    python scripts/verify_artifacts.py

Run it (a) after precompute, before pushing, and (b) on any machine
after git pull, before a demo. Exit code 0 (PASS) means the app will
read everything from data/derived/ and never fall back to slow
on-the-fly compute.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import artifacts
from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.ingest import load_corpus


def main() -> int:
    docs = [d for d in load_corpus(FILINGS_DIR, SAMPLE_DIR)
            if d.path.parent.name != "sample"]
    print(f"corpus: {len(docs)} real filings\n")
    ok = True

    # ---- sentiment ----------------------------------------------------
    art = artifacts.load_sentiment()
    missing = [d for d in docs
               if artifacts.text_key(d.full_text[:artifacts.SENT_CHARS])
               not in art]
    line = f"sentiment: {len(docs) - len(missing)}/{len(docs)} filings covered"
    if missing:
        ok = False
        print(f"[X] {line} — run: python scripts/precompute.py --sentiment --gpu")
    else:
        print(f"[OK] {line}")
    if art:
        backends = Counter(v.get("backend", "?") for v in art.values())
        print(f"     backends: {dict(backends)}")
        if "lexicon" in backends:
            ok = False
            print("     [X] some entries used the LEXICON fallback — different "
                  "scale from FinBERT. Fix the FinBERT install and rerun: "
                  "python scripts/precompute.py --sentiment --force")

    # ---- forward returns ----------------------------------------------
    ret = artifacts.load_returns()
    rows = [{"ticker": d.ticker, "date": d.date} for d in docs]
    if ret and artifacts.returns_cover(ret, rows):
        print(f"[OK] returns: all {len(rows)} filing dates covered "
              f"(prices generated {ret.get('generated_at')}, "
              f"windows {ret.get('windows')})")
    else:
        ok = False
        print("[X] returns: incomplete — run: "
              "python scripts/precompute.py --returns")

    # ---- diffs ---------------------------------------------------------
    dart = artifacts.load_diffs()
    by_t: dict[str, list] = {}
    for d in docs:
        by_t.setdefault(d.ticker, []).append(d)
    expected = []
    for tick, ds in by_t.items():
        ds = sorted(ds, key=lambda d: d.date)
        expected += [(tick, str(a.date), str(b.date))
                     for a, b in zip(ds, ds[1:])]
    have = sum(1 for tick, a, b in expected
               if artifacts._pair_key(a, b) in dart.get(tick, {}))
    if have == len(expected):
        print(f"[OK] diffs: all {len(expected)} adjacent pairs covered")
    else:
        ok = False
        print(f"[X] diffs: {have}/{len(expected)} adjacent pairs — run: "
              "python scripts/precompute.py --diffs "
              "(checkpointed; safe to interrupt and resume)")

    print()
    if ok:
        print("PASS — the app reads everything from data/derived/ with zero "
              "heavy compute.")
        return 0
    print("INCOMPLETE — the app still runs, but will fall back to slow "
          "on-the-fly compute for the gaps above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
