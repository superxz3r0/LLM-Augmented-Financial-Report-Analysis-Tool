from __future__ import annotations

import os
from urllib.parse import quote

_BUCKET = os.environ.get("FINSIGHT_S3_BUCKET", "")
_PREFIX = os.environ.get("FINSIGHT_S3_PREFIX", "filings/")
_EXPIRES = 3600          # 1h; capped anyway by the instance-role session

_s3 = None
_key_map: dict[str, str] | None = None


def _client():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


def _load_key_map() -> dict[str, str]:
    """stem -> full S3 key, built once by listing the bucket. Resolves the
    .txt/.htm/.html ambiguity without touching Chunk or the Chroma metadata."""
    global _key_map
    if _key_map is None:
        m: dict[str, str] = {}
        try:
            paginator = _client().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=_BUCKET, Prefix=_PREFIX):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    m[stem] = key
        except Exception as e:
            print(f"[s3links] key map unavailable ({e}); citations stay plain")
        _key_map = m
    return _key_map


def _presign(key: str) -> str:
    return _client().generate_presigned_url(
        "get_object", Params={"Bucket": _BUCKET, "Key": key}, ExpiresIn=_EXPIRES)


def _deep_url(url: str, chunk_text: str) -> str:
    """Append a text-fragment so supporting browsers jump to & highlight the
    passage. The fragment lives after '#': browsers never send it to S3, so
    the presigned signature is unaffected. Plain-text files just ignore it."""
    words = " ".join(chunk_text.split()[:8])
    return f"{url}#:~:text={quote(words)}"


def citation_link(chunk) -> str:
    """Markdown for one source: linked when the doc exists in S3, plain
    otherwise. Safe to call unconditionally."""
    if not _BUCKET:
        return chunk.citation
    key = _load_key_map().get(chunk.doc_id)
    if not key:
        return chunk.citation
    return f"[{chunk.citation}]({_deep_url(_presign(key), chunk.text)})"