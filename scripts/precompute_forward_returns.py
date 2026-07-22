"""Populate FinSight's portable forward-return cache.

The generated rows live in ``data/finsight.db`` and are reused by the
Streamlit app, including deployments that cannot contact Yahoo Finance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        action="append",
        help="Only populate this ticker (repeatable); SPY is always included.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 20],
        help="Forward-return windows in trading sessions (default: 5 20).",
    )
    return parser.parse_args()


def main() -> None:
    from finsight.config import FILINGS_DIR
    from finsight.ingest import load_corpus
    from finsight.returns import (
        MARKET_PROXY,
        MarketDataUnavailable,
        fetch_forward_returns,
    )

    args = parse_args()
    if not args.windows or any(window < 1 for window in args.windows):
        raise SystemExit("All forward-return windows must be positive")

    docs = load_corpus(FILINGS_DIR)
    dates_by_ticker: dict[str, set[str]] = {}
    for document in docs:
        dates_by_ticker.setdefault(document.ticker.upper(), set()).add(document.date)

    requested = {ticker.upper() for ticker in (args.ticker or dates_by_ticker)}
    unknown = requested - dates_by_ticker.keys()
    if unknown:
        raise SystemExit(f"No filing dates found for: {', '.join(sorted(unknown))}")

    all_dates = sorted({date for dates in dates_by_ticker.values() for date in dates})
    targets = [(MARKET_PROXY, all_dates)] + [
        (ticker, sorted(dates_by_ticker[ticker])) for ticker in sorted(requested)
    ]
    failures = []
    for ticker, dates in targets:
        try:
            values = fetch_forward_returns(ticker, dates, tuple(args.windows))
        except MarketDataUnavailable as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
            print(json.dumps(failures[-1]), flush=True)
            continue
        populated = sum(len(by_window) for by_window in values.values())
        print(
            json.dumps(
                {
                    "ticker": ticker,
                    "event_dates": len(dates),
                    "cached_forward_returns": populated,
                }
            ),
            flush=True,
        )

    if failures:
        raise SystemExit(f"Price cache failed for {len(failures)} ticker(s)")


if __name__ == "__main__":
    main()
