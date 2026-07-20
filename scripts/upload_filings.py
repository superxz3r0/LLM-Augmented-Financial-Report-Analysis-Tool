"""One-off uploader: put data/<dir> into s3://<bucket>/<dir>/ with correct
Content-Type. Many EDGAR .txt files actually contain HTML; without the right
Content-Type the browser downloads them instead of rendering inline.

Usage:  python scripts/upload_filings.py <bucket> filings
        python scripts/upload_filings.py <bucket> sample
"""
import sys
from pathlib import Path

import boto3

bucket, subdir = sys.argv[1], sys.argv[2]
src = Path("data") / subdir
s3 = boto3.client("s3")


def looks_html(p: Path) -> bool:
    head = p.read_text(encoding="utf-8", errors="ignore")[:5000]
    return "</" in head or "<html" in head.lower()


count = 0
for p in sorted(src.glob("*")):
    if p.suffix.lower() not in {".txt", ".htm", ".html"}:
        continue
    ctype = "text/html; charset=utf-8" if looks_html(p) else "text/plain; charset=utf-8"
    s3.upload_file(
        str(p), bucket, f"{subdir}/{p.name}",
        ExtraArgs={"ContentType": ctype, "ContentDisposition": "inline"},
    )
    print(f"  {p.name}  ->  {ctype}")
    count += 1
print(f"uploaded {count} files to s3://{bucket}/{subdir}/")