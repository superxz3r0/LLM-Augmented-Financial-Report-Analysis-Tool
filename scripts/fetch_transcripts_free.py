"""Fetch earnings-call transcripts into data/filings/ — FREE source.

Uses the `earningscall` library (pip install earningscall).
Without any API key it serves demo data: Apple (AAPL) and Microsoft (MSFT).
A key from https://earningscall.biz unlocks 5,000+ companies; the library
reads it from the EARNINGSCALL_API_KEY environment variable by itself.

Same on-disk contract as every other fetcher in this project:

    <TICKER>_TRANSCRIPT_<YYYY-MM-DD>.txt      into data/filings/

so ingest.load_corpus() picks the files up with zero code changes.

Format note: the basic (free) transcript level is flat text WITHOUT
speaker headers. ingest._split_transcript then falls back to a single
whole-call section — still fully chunkable and searchable, but chunks
lose per-speaker citations. Paid "enhanced" levels populate
prepared_remarks / questions_and_answers, which this script stitches
together with an explicit boundary line that ingest's Q&A detection
already understands.

Usage:
    python scripts/fetch_transcripts_free.py                    # AAPL+MSFT demo
    python scripts/fetch_transcripts_free.py --tickers NVDA --years 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "filings"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch earnings-call transcripts (free source) into data/filings/")
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"],
                    help="without an API key, demo access covers AAPL and MSFT only")
    ap.add_argument("--years", type=int, default=2,
                    help="how many years of calls per ticker (4 calls/year)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="pause between API calls")
    args = ap.parse_args()

    try:
        from earningscall import get_company
    except ImportError:
        print("pip install earningscall", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for ticker in args.tickers:
        company = get_company(ticker)
        if company is None:
            print(f"[skip] {ticker}: symbol not found")
            continue
        try:
            events = [e for e in company.events() if e.conference_date]
        except Exception as e:   # e.g. InsufficientApiAccessError without a key
            print(f"[skip] {ticker}: {type(e).__name__}: {e}")
            continue
        events.sort(key=lambda e: e.conference_date, reverse=True)

        for ev in events[: args.years * 4]:
            date = ev.conference_date.date().isoformat()
            out = DATA_DIR / f"{ticker}_TRANSCRIPT_{date}.txt"
            if out.exists():
                continue
            time.sleep(args.sleep)
            try:
                t = company.get_transcript(event=ev)
            except Exception as e:
                print(f"[skip] {ticker} {date}: {type(e).__name__}: {e}")
                continue
            if t is None:
                print(f"[skip] {ticker} {date}: no transcript available")
                continue
            # Prefer the PR/QA split when the plan provides it; write an
            # explicit boundary line so ingest labels Q&A chunks correctly.
            if t.prepared_remarks and t.questions_and_answers:
                text = (t.prepared_remarks.strip()
                        + "\n\nQuestions and Answers:\n\n"
                        + t.questions_and_answers.strip())
            else:
                text = t.text or ""
            if len(text) < 500:
                print(f"[skip] {ticker} {date}: transcript empty/too short")
                continue
            out.write_text(text, encoding="utf-8")
            n_saved += 1
            print(f"[ok] {out.name}")

    print(f"\nSaved {n_saved} transcripts to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())