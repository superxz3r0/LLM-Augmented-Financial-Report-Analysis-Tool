from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "filings"

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "GS",
    "JNJ", "PFE", "UNH", "XOM", "CVX", "WMT", "COST", "KO", "PEP", "DIS",
    "NFLX", "INTC", "AMD", "CRM",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch SEC EDGAR filings into data/filings/")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    try:
        from edgar import Company, set_identity
    except ImportError:
        print("pip install edgartools", file=sys.stderr)
        return 1

    set_identity(os.environ.get("EDGAR_IDENTITY", "FinSight student@ucdconnect.ie"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for ticker in args.tickers:
        try:
            company = Company(ticker)
        except Exception as e:
            print(f"[skip] {ticker}: {e}")
            continue
        for form in args.forms:
            per_year = 1 if form == "10-K" else 4
            filings = company.get_filings(form=form).head(args.years * per_year)
            for f in filings:
                date = str(f.filing_date)
                out = DATA_DIR / f"{ticker}_{form}_{date}.txt"
                if out.exists():
                    continue
                try:
                    out.write_text(f.text(), encoding="utf-8")
                    n_saved += 1
                    print(f"[ok] {out.name}")
                except Exception as e:
                    print(f"[skip] {ticker} {form} {date}: {e}")

    print(f"\nSaved {n_saved} filings to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
