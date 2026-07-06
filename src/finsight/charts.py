from __future__ import annotations

import base64
import html as html_lib
import json
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import Chunk
from .config import settings

_IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(
    r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_CHART_WORDS = {
    "chart": 5,
    "graph": 5,
    "performance": 3,
    "comparison": 2,
    "cumulative": 2,
    "return": 2,
    "revenue": 2,
    "growth": 2,
    "segment": 2,
    "trend": 2,
}
_NON_CHART_WORDS = ("logo", "signature", "seal", "cover", "icon")


@dataclass(frozen=True)
class ChartCandidate:
    image_url: str
    alt: str
    context: str
    width: int | None
    height: int | None
    score: int


def _clean_html_fragment(raw: str) -> str:
    text = _TAG.sub(" ", raw)
    return _SPACE.sub(" ", html_lib.unescape(text)).strip()


def _dimension(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def discover_chart_candidates(html: str, filing_url: str) -> list[ChartCandidate]:
    """Find likely chart images in a 10-K/10-Q HTML filing.

    SEC filings rarely label images consistently, so candidates are ranked using
    the image attributes, filename, nearby filing text, and basic dimensions.
    """
    found: list[ChartCandidate] = []
    for match in _IMG_TAG.finditer(html):
        attrs: dict[str, str] = {}
        for name, double, single, bare in _ATTR.findall(match.group()):
            attrs[name.lower()] = double or single or bare
        src = attrs.get("src")
        if not src or src.startswith("data:"):
            continue

        before = html[max(0, match.start() - 1800):match.start()]
        after = html[match.end():min(len(html), match.end() + 700)]
        context = _clean_html_fragment(before + " " + after)[-1600:]
        alt = attrs.get("alt", "")
        haystack = f"{src} {alt} {context}".lower()

        if any(word in f"{src} {alt}".lower() for word in _NON_CHART_WORDS):
            continue
        score = sum(weight for word, weight in _CHART_WORDS.items() if word in haystack)
        width, height = _dimension(attrs.get("width")), _dimension(attrs.get("height"))
        if width and height and width >= 400 and height >= 220:
            score += 2
        if re.search(r"(?:^|[_-])g\d+\.(?:png|jpe?g|gif)$", src, re.IGNORECASE):
            score += 1
        if score < 3:
            continue
        found.append(ChartCandidate(
            image_url=urllib.parse.urljoin(filing_url, src),
            alt=alt,
            context=context,
            width=width,
            height=height,
            score=score,
        ))

    best_by_url: dict[str, ChartCandidate] = {}
    for candidate in found:
        prior = best_by_url.get(candidate.image_url)
        if prior is None or candidate.score > prior.score:
            best_by_url[candidate.image_url] = candidate
    return sorted(best_by_url.values(), key=lambda c: (-c.score, c.image_url))


_VISION_PROMPT = """Analyze this financial-filing chart and return one JSON object.
Never infer a value that is not visible. Use null for unreadable values and mark
visually estimated values with "estimated": true. Preserve the chart's units.

Required keys:
- title: string
- chart_type: string (line, bar, area, pie, scatter, or other)
- summary: concise factual description
- x_axis: object with label, unit, and categories (array)
- y_axis: object with label, unit, minimum, and maximum
- series: array of objects; each has name and values, where each value has x, y,
  and estimated
- insights: array of factual comparisons or trends visible in the chart
- caveats: array describing estimation, missing labels, or historical-result notes

Filing context:
"""


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Vision model did not return a JSON object")
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Vision model response must be a JSON object")
    return payload


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("title", "chart_type", "summary", "x_axis", "y_axis", "series",
                "insights", "caveats")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Chart extraction is missing fields: {', '.join(missing)}")
    if not all(isinstance(payload[key], str) for key in ("title", "chart_type", "summary")):
        raise ValueError("title, chart_type, and summary must be strings")
    if not isinstance(payload["x_axis"], dict) or not isinstance(payload["y_axis"], dict):
        raise ValueError("x_axis and y_axis must be objects")
    if not isinstance(payload["series"], list) or not payload["series"]:
        raise ValueError("series must be a non-empty array")
    if not isinstance(payload["insights"], list) or not isinstance(payload["caveats"], list):
        raise ValueError("insights and caveats must be arrays")
    y_min = payload["y_axis"].get("minimum")
    y_max = payload["y_axis"].get("maximum")
    for series in payload["series"]:
        if not isinstance(series, dict) or not isinstance(series.get("name"), str):
            raise ValueError("Each series needs a string name")
        if not isinstance(series.get("values"), list):
            raise ValueError("Each series needs a values array")
        for point in series["values"]:
            if not isinstance(point, dict) or not {"x", "y", "estimated"} <= point.keys():
                raise ValueError("Each point needs x, y, and estimated fields")
            if not isinstance(point["estimated"], bool):
                raise ValueError("Each point's estimated field must be boolean")
            value = point["y"]
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError("Point values must be numbers or null")
            if isinstance(value, (int, float)):
                if isinstance(y_min, (int, float)) and value < y_min:
                    raise ValueError(f"Point value {value} is below the visible y-axis")
                if isinstance(y_max, (int, float)) and value > y_max:
                    raise ValueError(f"Point value {value} is above the visible y-axis")
    return payload


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Vision API returned HTTP {exc.code}: {detail}") from exc


def _image_data(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, encoded


def _extract_openai(path: Path, context: str) -> dict[str, Any]:
    mime, encoded = _image_data(path)
    model = os.environ.get("FINSIGHT_CHART_OPENAI_MODEL", settings.chart_openai_model)
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You extract grounded, machine-readable facts from financial charts.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT + context[:3000]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }
    response = _post_json(
        f"{os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        payload,
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )
    return _parse_json_response(response["choices"][0]["message"]["content"])


def _extract_gemini(path: Path, context: str) -> dict[str, Any]:
    mime, encoded = _image_data(path)
    model = os.environ.get("FINSIGHT_CHART_GEMINI_MODEL", settings.chart_gemini_model)
    key = urllib.parse.quote(os.environ["GEMINI_API_KEY"], safe="")
    payload = {
        "contents": [{
            "parts": [
                {"text": _VISION_PROMPT + context[:3000]},
                {"inline_data": {"mime_type": mime, "data": encoded}},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    response = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        payload,
        {},
    )
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json_response(text)


def extract_chart(path: Path, context: str = "", provider: str = "auto") -> dict[str, Any]:
    """Extract a chart into a validated retrieval schema using OpenAI or Gemini."""
    provider = provider.lower()
    if provider not in {"auto", "openai", "gemini"}:
        raise ValueError("provider must be auto, openai, or gemini")
    if provider in {"auto", "openai"} and os.environ.get("OPENAI_API_KEY"):
        return _validate_payload(_extract_openai(path, context))
    if provider in {"auto", "gemini"} and os.environ.get("GEMINI_API_KEY"):
        return _validate_payload(_extract_gemini(path, context))
    wanted = provider if provider != "auto" else "OpenAI or Gemini"
    raise RuntimeError(f"No {wanted} API key is configured for chart extraction")


def chart_record_to_text(record: dict[str, Any]) -> str:
    """Flatten structured chart data into a dense, retrieval-friendly text chunk."""
    lines = [
        "Content type: financial filing chart.",
        f"Company: {record.get('company') or record.get('ticker', '')}.",
        f"Filing: {record.get('ticker', '')} {record.get('form', '')} "
        f"for period {record.get('period_end') or record.get('date', '')}.",
        f"Chart title: {record.get('title', '')}.",
        f"Chart type: {record.get('chart_type', '')}.",
        f"Review status: {record.get('review_status', 'unreviewed vision extraction')}.",
        f"Summary: {record.get('summary', '')}",
    ]
    x_axis, y_axis = record.get("x_axis", {}), record.get("y_axis", {})
    lines.append(
        f"X-axis: {x_axis.get('label', '')}; unit: {x_axis.get('unit')}; "
        f"categories: {', '.join(str(x) for x in x_axis.get('categories', []))}."
    )
    lines.append(
        f"Y-axis: {y_axis.get('label', '')}; unit: {y_axis.get('unit')}; "
        f"range: {y_axis.get('minimum')} to {y_axis.get('maximum')}."
    )
    for series in record.get("series", []):
        values = []
        for point in series.get("values", []):
            marker = " (estimated)" if point.get("estimated") else ""
            values.append(f"{point.get('x')}: {point.get('y')}{marker}")
        lines.append(f"Series {series.get('name', '')}: " + "; ".join(values) + ".")
    if record.get("insights"):
        lines.append("Visible findings: " + " | ".join(map(str, record["insights"])) + ".")
    if record.get("caveats"):
        lines.append("Caveats: " + " | ".join(map(str, record["caveats"])) + ".")
    return "\n".join(lines)


def chart_chunk_from_file(path: Path) -> Chunk:
    record = json.loads(path.read_text(encoding="utf-8"))
    image_name = record.get("image_file")
    image_path = (path.parent / image_name).resolve() if image_name else None
    if image_path and not image_path.exists():
        image_path = None
    ticker = str(record["ticker"]).upper()
    form = str(record["form"]).upper()
    date = str(record.get("period_end") or record.get("date"))
    chart_id = str(record.get("chart_id") or path.stem.removesuffix(".chart"))
    return Chunk(
        chunk_id=f"{ticker}_{form}_{date}#chart-{chart_id}",
        doc_id=f"{ticker}_{form}_{date}",
        ticker=ticker,
        form=form,
        date=date,
        item=str(record.get("item", "unknown")),
        section_title=str(record.get("title", "Filing chart")),
        text=chart_record_to_text(record),
        content_type="chart",
        asset_path=image_path,
        source_url=str(record.get("filing_url", "")),
    )


def load_chart_chunks(*dirs: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[Path] = set()
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.chart.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                chunks.append(chart_chunk_from_file(path))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[charts] skipping invalid sidecar {path}: {exc}")
    return chunks


def write_chart_sidecar(
    image_path: Path,
    extraction: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    record = {**metadata, **_validate_payload(extraction)}
    record["schema_version"] = 1
    record["image_file"] = image_path.name
    record.setdefault("review_status", "unreviewed vision extraction")
    out = image_path.with_suffix(".chart.json")
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
