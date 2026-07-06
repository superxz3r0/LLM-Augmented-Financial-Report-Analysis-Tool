from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.charts import (  # noqa: E402
    ChartCandidate,
    discover_chart_candidates,
    extract_chart,
    write_chart_sidecar,
)
from finsight.config import CHARTS_DIR  # noqa: E402

USER_AGENT = os.environ.get(
    "EDGAR_IDENTITY",
    "FinSight chart ingestion student@ucdconnect.ie",
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        return response.read()


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _download_candidate(candidate: ChartCandidate, output: Path, prefix: str, number: int) -> Path:
    suffix = Path(urllib.parse.urlparse(candidate.image_url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        suffix = ".jpg"
    target = output / f"{prefix}_chart-{number}{suffix}"
    target.write_bytes(_read_url(candidate.image_url))
    return target


def _metadata(args, image_url: str, chart_id: str, context: str) -> dict:
    return {
        "chart_id": chart_id,
        "ticker": args.ticker.upper(),
        "company": args.company or args.ticker.upper(),
        "form": args.form.upper(),
        "period_end": args.period_end,
        "filing_date": args.filing_date or "",
        "item": args.item,
        "filing_url": args.source_filing_url or args.filing_url or "",
        "image_url": image_url,
        "filing_context": context,
        "extraction_provider": args.provider,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover charts in a 10-K/10-Q, extract them with vision, and create RAG sidecars.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--filing-url", help="SEC filing HTML URL")
    source.add_argument("--local-image", type=Path, help="Already-downloaded chart image")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company")
    parser.add_argument("--form", required=True, choices=["10-K", "10-Q", "10-k", "10-q"])
    parser.add_argument("--period-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--filing-date", help="YYYY-MM-DD")
    parser.add_argument("--source-filing-url", help="Original filing URL for a local image")
    parser.add_argument("--source-image-url", help="Original image URL for a local image")
    parser.add_argument("--item", default="5")
    parser.add_argument("--context", default="", help="Optional nearby filing text")
    parser.add_argument("--provider", choices=["auto", "openai", "gemini"], default="auto")
    parser.add_argument("--max-charts", type=int, default=5)
    parser.add_argument("--min-score", type=int, default=5)
    parser.add_argument("--output", type=Path, default=CHARTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="List discovered images without API calls")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = _safe_stem(f"{args.ticker.upper()}_{args.form.upper()}_{args.period_end}")

    if args.local_image:
        if not args.local_image.exists():
            parser.error(f"Image not found: {args.local_image}")
        candidates = [ChartCandidate(
            image_url=args.local_image.resolve().as_uri(),
            alt="",
            context=args.context,
            width=None,
            height=None,
            score=100,
        )]
    else:
        filing_html = _read_url(args.filing_url).decode("utf-8", errors="replace")
        candidates = [
            candidate for candidate in discover_chart_candidates(filing_html, args.filing_url)
            if candidate.score >= args.min_score
        ][:args.max_charts]

    if not candidates:
        print("No likely charts found. Try lowering --min-score.")
        return 2

    if args.dry_run:
        for candidate in candidates:
            print(f"score={candidate.score:2d}  {candidate.image_url}")
            print(f"  context: {candidate.context[:240]}")
        return 0

    for number, candidate in enumerate(candidates, 1):
        if args.local_image:
            image_path = args.output / args.local_image.name
            if image_path.resolve() != args.local_image.resolve():
                shutil.copy2(args.local_image, image_path)
        else:
            image_path = _download_candidate(candidate, args.output, prefix, number)

        context = args.context or candidate.context
        print(f"[vision] extracting {image_path.name} with {args.provider}")
        extraction = extract_chart(image_path, context=context, provider=args.provider)
        sidecar = write_chart_sidecar(
            image_path,
            extraction,
            _metadata(args, args.source_image_url or candidate.image_url, str(number), context),
        )
        print(f"[ok] {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
