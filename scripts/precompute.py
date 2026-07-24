"""Precompute every expensive artifact the app reads.

    python scripts/precompute.py --all              # pure CPU, runs anywhere
    python scripts/precompute.py --sentiment --gpu  # sentiment on a Verda burst
    python scripts/precompute.py --returns          # network only (yfinance)
    python scripts/precompute.py --diffs

Artifacts land in data/derived/ (see finsight/artifacts.py for schemas).
Then sync that one directory to wherever the app runs:

    rsync -avz data/derived/ user@host:/path/to/finsight/data/derived/

Notes
-----
* Sentence splitting for sentiment reuses finsight/sentiment.py's own
  regex + filters, and the model name comes from settings.finbert_model,
  so the GPU path can never drift from what the app computes on CPU.
* forward_returns.json goes stale as time passes (recent filings gain
  tradeable 5/20-day windows, prices extend) — rerun --returns before a
  demo or on a schedule.
* Defaults score the FULL filing text, every sentence — the signal
  definition lives in artifacts.SENT_CHARS / SENT_MAX_SENTENCES.
  --chars / --max-sentences override for experiments, but they change
  cache keys and score values: keep the app-side constants in lockstep.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finsight import artifacts
from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.ingest import load_corpus


def _real_docs():
    return [d for d in load_corpus(FILINGS_DIR, SAMPLE_DIR)
            if d.path.parent.name != "sample"]


def _split_sentences(text: str, max_sentences: int) -> list[str]:
    # Reuse the app's own splitter so CPU and GPU inputs are identical.
    from finsight.sentiment import _SENT_SPLIT
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]
    return sents[:max_sentences]


# ------------------------------------------------------------- sentiment

def precompute_sentiment(docs, chars: int, max_sentences: int,
                         use_gpu: bool, force: bool) -> None:
    art = {} if force else artifacts.load_sentiment()
    todo = []
    for d in docs:
        text = d.full_text if chars == 0 else d.full_text[:chars]
        k = artifacts.text_key(text)
        if k not in art:
            todo.append((d, text, k))
    if not todo:
        print(f"sentiment: {len(art)} entries, nothing to do")
        return

    print(f"sentiment: scoring {len(todo)} of {len(docs)} filings "
          f"({'Verda GPU burst' if use_gpu else 'local CPU'})")

    if use_gpu:
        from finsight import gpu_burst
        from finsight.config import settings
        payload = {
            "model": settings.finbert_model,
            "docs": [{"id": k, "sentences": _split_sentences(t, max_sentences)}
                     for _d, t, k in todo],
        }
        n_sents = sum(len(x["sentences"]) for x in payload["docs"])
        print(f"  {n_sents:,} sentences -> GPU")
        scored = gpu_burst.burst_score(payload)      # {id: {score, n_sentences}}
        for d, _t, k in todo:
            art[k] = {**scored[k], "backend": "finbert-gpu",
                      "ticker": d.ticker, "date": str(d.date), "form": d.form}
    else:
        from finsight import sentiment
        backends: set[str] = set()
        for i, (d, t, k) in enumerate(todo, 1):
            r = sentiment.score_text(t, max_sentences=max_sentences)
            backends.add(r.backend)
            art[k] = {"score": r.score, "n_sentences": r.n_sentences,
                      "backend": r.backend,
                      "ticker": d.ticker, "date": str(d.date), "form": d.form}
            print(f"\r  {i}/{len(todo)}", end="", flush=True)
        print()
        if "lexicon" in backends:
            print("  !! WARNING: some filings silently fell back to the "
                  "LEXICON scorer (is FinBERT installed?). Lexicon and "
                  "FinBERT scores are on different scales — do NOT mix them "
                  "in one regression. Fix the install and rerun with --force.")

    artifacts.save_sentiment(art)
    print(f"sentiment -> {artifacts.SENTIMENT_PATH} ({len(art)} entries)")


# ------------------------------------------------------- forward returns

def precompute_returns(docs, windows=(5, 20)) -> None:
    from finsight.returns import MARKET_PROXY, fetch_forward_returns

    by_ticker: dict[str, list] = {}
    for d in docs:
        by_ticker.setdefault(d.ticker, []).append(d.date)
    all_dates = sorted({d.date for d in docs})

    print(f"returns: fetching {len(by_ticker)} tickers + {MARKET_PROXY} "
          "via yfinance…")
    art = {"generated_at": time.strftime("%Y-%m-%d %H:%M"),
           "windows": list(windows), "market": {}, "by_ticker": {}}

    mkt = fetch_forward_returns(MARKET_PROXY, all_dates, windows)
    art["market"] = {str(dt): {str(w): r for w, r in m.items()}
                     for dt, m in mkt.items()}

    for i, (tick, dates) in enumerate(sorted(by_ticker.items()), 1):
        fr = fetch_forward_returns(tick, sorted(set(dates)), windows)
        art["by_ticker"][tick] = {str(dt): {str(w): r for w, r in m.items()}
                                  for dt, m in fr.items()}
        print(f"\r  {i}/{len(by_ticker)} {tick}    ", end="", flush=True)
    print()

    artifacts.save_returns(art)
    print(f"returns -> {artifacts.RETURNS_PATH}")


# ----------------------------------------------------------------- diffs

def precompute_diffs(docs, force: bool) -> None:
    from finsight import diff as diff_mod

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("  !! WARNING: sentence-transformers is not installed — diffs "
              "will use the jaccard fallback, whose similarity scale and "
              "thresholds differ from the embeddings backend the app uses "
              "when available. For consistent artifacts:\n"
              "       pip install sentence-transformers")

    art = {} if force else artifacts.load_diffs()
    by_t: dict[str, list] = {}
    for d in docs:
        by_t.setdefault(d.ticker, []).append(d)

    pairs = []
    for tick, ds in by_t.items():
        ds = sorted(ds, key=lambda d: d.date)
        for a, b in zip(ds, ds[1:]):                 # adjacent filings only
            if artifacts._pair_key(str(a.date), str(b.date)) not in art.get(tick, {}):
                pairs.append((tick, a, b))
    if not pairs:
        print("diffs: nothing to do")
        return

    print(f"diffs: {len(pairs)} adjacent filing pairs")
    for i, (tick, a, b) in enumerate(pairs, 1):
        items = diff_mod.diff_documents(a, b)
        art.setdefault(tick, {})[artifacts._pair_key(str(a.date), str(b.date))] = {
            "old_form": a.form, "new_form": b.form,
            "items": [{"kind": it.kind, "item": it.item,
                       "similarity": it.similarity,
                       "old_text": it.old_text, "new_text": it.new_text}
                      for it in items],
        }
        print(f"\r  {i}/{len(pairs)} {tick} {a.date} -> {b.date}    ",
              end="", flush=True)
        if i % 10 == 0:
            artifacts.save_diffs(art)                # checkpoint long runs
    print()

    artifacts.save_diffs(art)
    print(f"diffs -> {artifacts.DIFFS_PATH}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentiment", action="store_true")
    ap.add_argument("--returns", action="store_true")
    ap.add_argument("--diffs", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="run sentiment inference on an ephemeral Verda GPU")
    ap.add_argument("--chars", type=int, default=None,
                    help="chars of each filing to score (default: FULL text, "
                         "per artifacts.SENT_CHARS; 0 also means full). "
                         "Changing this changes cache keys — keep in lockstep "
                         "with artifacts.SENT_CHARS, which the app fallback "
                         "uses.")
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="cap on sentences per filing (default: no cap — "
                         "score every sentence; mirror any change in "
                         "artifacts.SENT_MAX_SENTENCES)")
    ap.add_argument("--force", action="store_true",
                    help="recompute entries that already exist")
    a = ap.parse_args()

    if not (a.sentiment or a.returns or a.diffs or a.all):
        ap.error("pick at least one of --sentiment / --returns / --diffs / --all")

    docs = _real_docs()
    print(f"{len(docs)} real filings loaded")
    chars = artifacts.SENT_CHARS if a.chars is None else a.chars

    if a.all or a.sentiment:
        precompute_sentiment(docs, chars, a.max_sentences, a.gpu, a.force)
    if a.all or a.returns:
        precompute_returns(docs)
    if a.all or a.diffs:
        precompute_diffs(docs, a.force)

    print("\nDone. Sync artifacts to the app host, e.g.:\n"
          "  rsync -avz data/derived/ user@host:/path/to/finsight/data/derived/")


if __name__ == "__main__":
    main()
