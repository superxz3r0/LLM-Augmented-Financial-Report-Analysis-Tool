from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .llm import (
    LLMCallError,
    api_key_status,
    call_with_retries,
    gemini_error,
    openai_error,
)

SYSTEM_PROMPT = """You are a financial filings analyst. Answer the question using ONLY the numbered context passages provided. Cite passages inline as [1], [2] etc. after each claim. If the context does not contain the answer, say so explicitly; do not guess. Keep answers under 200 words."""

_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_FORM_10K = re.compile(r"\b10\s*-\s*k\b", re.IGNORECASE)
_FORM_10Q = re.compile(r"\b10\s*-\s*q\b", re.IGNORECASE)
_ITEM = re.compile(r"\bitem\s+(\d{1,2}[ab]?)\b", re.IGNORECASE)
_FISCAL_YEAR = re.compile(r"\bfiscal\s+(20\d{2})\b", re.IGNORECASE)
_FINANCIAL_VALUE = re.compile(
    r"(\$[\d,.]+|\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s+"
    r"(?:million|billion|trillion)\b)",
    re.IGNORECASE,
)

_TOPIC_PROFILES: dict[str, list[tuple[str, str]]] = {
    "risk": [
        ("competition", r"\bcompet"),
        ("regulation or legal proceedings", r"\b(regulat|legal|litigation|compliance|law)\b"),
        ("supply chain or manufacturing", r"\b(supply|supplier|inventory|component|manufactur)\b"),
        ("cybersecurity or data risk", r"\b(cyber|security|data breach|privacy)\b"),
        ("customer demand or sales channels", r"\b(demand|customer|consumer|reseller|sales channel)\b"),
        ("liquidity or capital risk", r"\b(liquidity|cash|capital|debt|credit)\b"),
        ("product or platform risk", r"\b(product|service|platform|technology|innovation)\b"),
    ],
    "mda": [
        ("revenue or net sales", r"\b(revenue|net sales)\b"),
        ("margin or profitability", r"\b(margin|profitability|operating income|gross profit)\b"),
        ("liquidity and capital resources", r"\b(liquidity|cash|capital resources|debt|credit)\b"),
        ("expenses or costs", r"\b(expense|cost|spend|spending)\b"),
        ("growth or demand", r"\b(growth|grew|increase|decrease|decline|demand)\b"),
        ("guidance or outlook", r"\b(guidance|outlook|expect|forecast)\b"),
    ],
    "transcript": [
        ("revenue or net sales", r"\b(revenue|net sales)\b"),
        ("guidance or outlook", r"\b(guidance|outlook|expect|forecast)\b"),
        ("margin or profitability", r"\b(margin|profitability|operating income|gross profit)\b"),
        ("growth or demand", r"\b(growth|grew|increase|decrease|decline|demand)\b"),
        ("cash or capital allocation", r"\b(cash|capital allocation|dividend|buyback|repurchase)\b"),
        ("AI, cloud, or data center demand", r"\b(AI|cloud|data center|datacenter)\b"),
    ],
}

_TOPIC_ORDER = {
    "revenue or net sales": 10,
    "margin or profitability": 9,
    "customer demand or sales channels": 8,
    "growth or demand": 8,
    "liquidity and capital resources": 7,
    "liquidity or capital risk": 7,
    "regulation or legal proceedings": 6,
    "supply chain or manufacturing": 5,
    "competition": 4,
    "expenses or costs": 4,
    "guidance or outlook": 3,
    "cash or capital allocation": 3,
    "AI, cloud, or data center demand": 3,
    "product or platform risk": 2,
    "cybersecurity or data risk": 1,
}


@dataclass
class RagAnswer:
    text: str
    sources: list
    backend: str
    failure_category: str | None = None
    error_message: str | None = None
    retrieval_sources: list | None = None


def _format_context(hits) -> str:
    return "\n\n".join(f"[{i + 1}] ({c.citation})\n{c.text}" for i, (c, _) in enumerate(hits))


def _openai_generate(question: str, context: str) -> str:
    status = api_key_status("openai", "OPENAI_API_KEY", aliases=("OPENAI_KEY",))
    if not status.valid:
        raise LLMCallError("openai", "configuration_failure", status.message)

    def request() -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            timeout=30,
        )
        return resp.choices[0].message.content or ""

    return call_with_retries(request, openai_error)


def _gemini_generate(question: str, context: str) -> str:
    status = api_key_status("gemini", "GEMINI_API_KEY")
    if not status.valid:
        raise LLMCallError("gemini", "configuration_failure", status.message)

    def request() -> str:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(settings.gemini_model, system_instruction=SYSTEM_PROMPT)
        resp = model.generate_content(
            f"Context:\n{context}\n\nQuestion: {question}",
            request_options={"timeout": 30},
        )
        return resp.text or ""

    return call_with_retries(request, gemini_error)


def extract_query_filters(question: str) -> dict[str, str | list[str]]:
    q = question.lower()
    filters: dict[str, str | list[str]] = {}
    dates = _DATE.findall(question)
    if dates:
        filters["date"] = sorted(set(dates))

    forms = []
    if _FORM_10K.search(question):
        forms.append("10-K")
    if _FORM_10Q.search(question):
        forms.append("10-Q")
    if "transcript" in q or "earnings call" in q:
        forms.append("TRANSCRIPT")
    if forms:
        filters["form"] = sorted(set(forms))

    items = [m.group(1).upper() for m in _ITEM.finditer(question)]
    if not items and "md&a" in q and "10-q" in q:
        items.append("2")
    if not items and "risk factor" in q and "10-k" in q:
        items.append("1A")
    if items:
        filters["item"] = sorted(set(items))
    return filters


def _merge_filters(inferred: dict[str, Any], explicit: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(inferred)
    if explicit:
        for key, value in explicit.items():
            if value not in (None, "", [], ()):
                merged[key] = value
    return merged


def _search(index, question: str, k: int, ticker: str | None, filters: dict[str, Any]):
    try:
        return index.search(question, k, ticker=ticker, **filters)
    except TypeError:
        if ticker:
            try:
                return index.search(question, k, ticker=ticker)
            except TypeError:
                pass
        return index.search(question, k)


def _retrieve(index, question: str, k: int, ticker: str | None,
              filters: dict[str, Any]) -> list:
    search_pool = max(k * 12, 60)
    section_chunks = _section_scan_chunks(index, ticker, filters)
    candidates = [(chunk, 0.0) for chunk in section_chunks]
    candidates.extend(_search(index, question, search_pool, ticker, filters))

    return _rank_candidates(question, candidates, filters)


def _section_scan_chunks(index, ticker: str | None, filters: dict[str, Any]) -> list:
    if not _has_precise_section_filter(filters):
        return []
    getter = getattr(index, "filtered_chunks", None)
    if getter is None:
        return []
    try:
        chunks = getter(ticker=ticker, **filters)
    except TypeError:
        return []
    # Exact filing/section scans are normally small. Keep a hard ceiling so a
    # malformed broad filter cannot turn retrieval into a corpus dump.
    if len(chunks) > 3000:
        return []
    return chunks


def _has_precise_section_filter(filters: dict[str, Any]) -> bool:
    return bool(filters.get("date") and filters.get("form") and filters.get("item"))


def _rank_candidates(question: str, candidates: list, filters: dict[str, Any]) -> list:
    profile = _profile_for_filters(question, filters)
    topic, pattern = _infer_topic(question, profile)
    by_id: dict[str, tuple[Any, float, float, int, int]] = {}
    for rank, (chunk, base_score) in enumerate(candidates, 1):
        chunk_id = getattr(chunk, "chunk_id", str(id(chunk)))
        evidence_score = _evidence_score(chunk.text, profile, topic, pattern)
        query_score = _query_overlap(question, chunk.text)
        evidence_len = _concise_evidence_length(chunk.text, pattern)
        if topic and evidence_len < 180:
            evidence_score = -50.0
        score = evidence_score + (0.05 * query_score) + (0.02 / rank) + (0.01 * float(base_score))
        current = by_id.get(chunk_id)
        if current is None or (evidence_score, -evidence_len, score) > (
            current[1], -current[2], current[4]
        ):
            by_id[chunk_id] = (chunk, evidence_score, evidence_len, rank, score)
    ordered = sorted(
        by_id.values(),
        key=lambda item: (-item[1], item[2], item[3]),
    )
    return [(chunk, score) for chunk, _, _, _, score in ordered]


def _profile_for_filters(question: str, filters: dict[str, Any]) -> str:
    q = question.lower()
    forms = {str(v).upper() for v in _filter_values(filters.get("form"))}
    items = {str(v).upper() for v in _filter_values(filters.get("item"))}
    if "TRANSCRIPT" in forms or "transcript" in q or "earnings call" in q:
        return "transcript"
    if "10-Q" in forms or "2" in items or "md&a" in q:
        return "mda"
    return "risk"


def _infer_topic(question: str, profile: str) -> tuple[str | None, str | None]:
    q = question.lower()
    for topic, pattern in _TOPIC_PROFILES.get(profile, []):
        if topic.lower() in q:
            return topic, pattern
    for topics in _TOPIC_PROFILES.values():
        for topic, pattern in topics:
            if topic.lower() in q:
                return topic, pattern
    return None, None


def _evidence_score(text: str, profile: str, topic: str | None,
                    pattern: str | None) -> float:
    first_topic = _first_matching_topic(text, profile)
    if topic and first_topic != topic:
        return -50.0
    score = float(_TOPIC_ORDER.get(topic or "", 0))
    score += min(len(text) // 200, 5)
    if profile in {"mda", "transcript"} and _FINANCIAL_VALUE.search(text):
        score += 8
    if _looks_like_boilerplate(text):
        score -= 20
    return score


def _first_matching_topic(text: str, profile: str) -> str | None:
    for topic, pattern in _TOPIC_PROFILES.get(profile, []):
        if re.search(pattern, text, flags=re.IGNORECASE):
            return topic
    return None


def _looks_like_boilerplate(text: str) -> bool:
    low = text.lower()
    return (
        "table of contents" in low
        or "references to website urls" in low
        or "inactive textual references" in low
        or "forward-looking statements" in low
    )


def _concise_evidence_length(text: str, pattern: str | None) -> int:
    return len(_concise_evidence_text(text, pattern))


def _snippet_for_question(question: str, text: str, filters: dict[str, Any]) -> str:
    profile = _profile_for_filters(question, filters)
    _, pattern = _infer_topic(question, profile)
    return _concise_evidence_text(text, pattern)


def _concise_evidence_text(text: str, pattern: str | None) -> str:
    if not pattern:
        return text[:700].rsplit(" ", 1)[0].strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
        if 60 <= len(s.strip()) <= 1200
    ]
    if not sentences:
        return text[:900].strip()
    match_index = 0
    for index, sentence in enumerate(sentences):
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            match_index = index
            break
    picked = 0
    out: list[str] = []
    total = 0
    for sentence in sentences[match_index:]:
        if total + len(sentence) > 900 and picked:
            break
        out.append(sentence)
        total += len(sentence) + 1
        picked += 1
        if total >= 240 and picked >= 2:
            break
    return " ".join(out).strip()


def _query_overlap(question: str, text: str) -> float:
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "into", "what",
        "does", "did", "how", "its", "are", "was", "were", "has", "have",
        "according", "item", "filing", "filings", "say", "about", "what",
    }
    q = {t for t in re.findall(r"[a-z0-9][a-z0-9-]*", question.lower()) if t not in stop}
    if not q:
        return 0.0
    t = set(re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()))
    return len(q & t) / len(q)


def _filter_values(value: Any) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _token_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _dedupe(hits: list, k: int) -> list:
    kept: list = []
    for chunk, score in hits:
        if any(_token_jaccard(chunk.text, kc.text) > 0.85 for kc, _ in kept):
            continue
        kept.append((chunk, score))
        if len(kept) >= k:
            break
    return kept


def _should_abstain(question: str, hits: list) -> bool:
    q = question.lower()
    fiscal_years = set(_FISCAL_YEAR.findall(question))
    if not fiscal_years or "guidance" not in q:
        return False
    searchable_context = " ".join(
        getattr(c, "text", "") + " " + getattr(c, "date", "") for c, _ in hits
    )
    return not any(year in searchable_context for year in fiscal_years)


def _extractive_answer(question: str, hits: list,
                       filters: dict[str, Any] | None = None,
                       failure: LLMCallError | None = None) -> RagAnswer:
    retrieval_hits = list(hits)
    hits = _select_extractive_hits(question, hits, filters or {})
    if _should_abstain(question, hits):
        years = ", ".join(sorted(set(_FISCAL_YEAR.findall(question)))) or "the requested period"
        text = f"The retrieved filings do not provide revenue guidance for fiscal {years}."
        return RagAnswer(
            text, [], "extractive",
            failure.category if failure else None,
            str(failure) if failure else None,
            retrieval_hits,
        )

    lines = []
    if failure:
        lines.append(
            f"(Extractive fallback after {failure.provider} {failure.category}: {failure})\n"
        )
    elif not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        lines.append("(Extractive mode; set OPENAI_API_KEY or GEMINI_API_KEY for generated answers.)\n")

    for i, (c, _) in enumerate(hits, 1):
        snippet = _snippet_for_question(question, c.text, filters or {})
        lines.append(f"[{i}] {snippet}")
    return RagAnswer(
        "\n\n".join(lines), hits, "extractive",
        failure.category if failure else None,
        str(failure) if failure else None,
        retrieval_hits,
    )


def _select_extractive_hits(question: str, hits: list,
                            filters: dict[str, Any]) -> list:
    if not hits:
        return hits
    dates = [str(value) for value in _filter_values(filters.get("date"))]
    q = question.lower()
    if len(dates) > 1 or "compar" in q:
        by_date: dict[str, tuple[Any, float]] = {}
        for hit in hits:
            chunk, _ = hit
            date = getattr(chunk, "date", "")
            if date in dates and date not in by_date:
                by_date[date] = hit
        selected = [by_date[date] for date in dates if date in by_date]
        return selected or hits[: min(2, len(hits))]
    if _has_precise_section_filter(filters):
        top_score = float(hits[0][1])
        selected = [hits[0]]
        for hit in hits[1:]:
            if len(selected) >= 3:
                break
            if top_score - float(hit[1]) <= 1.0:
                selected.append(hit)
        return selected
    return hits[: min(3, len(hits))]


def retrieve_contexts(index, question: str, ticker: str | None = None,
                      k: int | None = None,
                      filters: dict[str, Any] | None = None) -> tuple[list, dict[str, Any]]:
    k = k or settings.top_k
    search_filters = _merge_filters(extract_query_filters(question), filters)
    hits = _dedupe(_retrieve(index, question, k, ticker, search_filters), k)
    return hits, search_filters


def answer(index, question: str, ticker: str | None = None, k: int | None = None,
           history: list[tuple[str, str]] | None = None,
           filters: dict[str, Any] | None = None) -> RagAnswer:
    hits, search_filters = retrieve_contexts(
        index, question, ticker=ticker, k=k, filters=filters
    )
    if not hits:
        return RagAnswer("No relevant passages found in the corpus.", [], "none")

    context = _format_context(hits)
    if history:
        recent = history[-2:]
        convo = "\n".join(f"Q: {q}\nA: {a[:400]}" for q, a in recent)
        question = f"(Recent conversation for context:\n{convo})\n\nCurrent question: {question}"

    last_failure: LLMCallError | None = None
    openai_status = api_key_status("openai", "OPENAI_API_KEY", aliases=("OPENAI_KEY",))
    if openai_status.configured:
        try:
            return RagAnswer(_openai_generate(question, context), hits, "openai")
        except LLMCallError as exc:
            last_failure = exc
            print(f"[rag] OpenAI {exc.category}: {exc}")

    gemini_status = api_key_status("gemini", "GEMINI_API_KEY")
    if gemini_status.configured:
        try:
            return RagAnswer(_gemini_generate(question, context), hits, "gemini")
        except LLMCallError as exc:
            last_failure = exc
            print(f"[rag] Gemini {exc.category}: {exc}")

    return _extractive_answer(question, hits, search_filters, last_failure)
