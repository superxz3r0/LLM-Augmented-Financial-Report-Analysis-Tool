from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .ingest import Document
from . import sentiment

_MONEY = r"(?:\$|US\$|USD\s?)?\s?\d[\d,.]*\s?(?:billion|million|bn|m\b)"
_REV_GUIDE = re.compile(
    rf"(?:expect|anticipate|project|guidance|outlook)[^.]{{0,160}}?"
    rf"(?:revenue|net sales|total sales)[^.]{{0,160}}?({_MONEY}(?:\s?(?:to|and|-|–)\s?{_MONEY})?)",
    re.IGNORECASE,
)
_CAPEX_GUIDE = re.compile(
    rf"(?:capital expenditures?|capex)[^.]{{0,200}}?({_MONEY}(?:\s?(?:to|and|-|–)\s?{_MONEY})?)",
    re.IGNORECASE,
)
_RISK_HEADING = re.compile(
    r"^(?=.{15,160}$)[A-Z][^!?\n]*?(?:risk|adversely|could|may|uncertain|depend|harm|expose|fail|litigation)[^!?\n]*?\.?$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class SignalSet:
    doc_id: str
    revenue_guidance: list[str]
    capex_guidance: list[str]
    risk_factor_count: int
    sentiment_score: float
    sentiment_backend: str
    method: str = "regex"
    xcheck_agree: bool | None = None

    def as_dict(self) -> dict:
        return asdict(self)


_LLM_PROMPT = """You extract guidance from an SEC filing's MD&A section.
Respond with ONLY a JSON object, no markdown fences, with exactly these keys:
{"revenue_guidance": [<verbatim quoted figures/ranges for guided revenue, max 5>],
 "capex_guidance": [<verbatim quoted figures/ranges for guided capital expenditures, max 5>]}
Quote figures verbatim from the text (e.g. "$2.7 billion to $2.9 billion").
Use [] when no guidance is stated. Do not infer or compute numbers."""


def _validate_llm_payload(raw: str) -> dict | None:
    import json as _json
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        data = _json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for key in ("revenue_guidance", "capex_guidance"):
        v = data.get(key)
        if not isinstance(v, list) or len(v) > 5 \
                or not all(isinstance(x, str) and 0 < len(x) < 200 for x in v):
            return None
        out[key] = v
    return out


def _llm_guidance(mda: str) -> dict | None:
    import os
    from .llm import (
        LLMCallError,
        api_key_status,
        call_with_retries,
        gemini_error,
        openai_error,
    )
    text = mda[:12000]
    try:
        if api_key_status("gemini", "GEMINI_API_KEY").valid:
            def gemini_request() -> dict | None:
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                model = genai.GenerativeModel(settings_model(), system_instruction=_LLM_PROMPT)
                return _validate_llm_payload(model.generate_content(text).text)
            return call_with_retries(gemini_request, gemini_error)
        if api_key_status("openai", "OPENAI_API_KEY", aliases=("OPENAI_KEY",)).valid:
            def openai_request() -> dict | None:
                from openai import OpenAI
                from .config import settings as _s
                resp = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
                    model=_s.openai_model, temperature=0,
                    messages=[{"role": "system", "content": _LLM_PROMPT},
                              {"role": "user", "content": text}],
                    timeout=30,
                )
                return _validate_llm_payload(resp.choices[0].message.content or "")
            return call_with_retries(openai_request, openai_error)
    except LLMCallError as exc:
        print(f"[extract] {exc.provider} {exc.category}: {exc}")
    return None


def settings_model() -> str:
    from .config import settings
    return settings.gemini_model


def _regex_guidance(mda: str) -> dict:
    rev = list(dict.fromkeys(m.group(1).strip() for m in _REV_GUIDE.finditer(mda)))[:5]
    capex = list(dict.fromkeys(m.group(1).strip() for m in _CAPEX_GUIDE.finditer(mda)))[:5]
    return {"revenue_guidance": rev, "capex_guidance": capex}


def _numbers(strings: list[str]) -> set[str]:
    return {n for s in strings for n in re.findall(r"\d[\d,.]*", s)}


def extract_signals(doc: Document, use_llm: bool = True) -> SignalSet:
    mda = _section_text(doc, ("7", "2"))
    risk = _section_text(doc, ("1A",))

    regex_g = _regex_guidance(mda)
    llm_g = _llm_guidance(mda) if use_llm else None

    if llm_g is not None:
        guidance, method = llm_g, "llm+regex-xcheck"
        agree = _numbers(regex_g["revenue_guidance"] + regex_g["capex_guidance"]) \
            <= _numbers(llm_g["revenue_guidance"] + llm_g["capex_guidance"]) \
            or not any(regex_g.values())
    else:
        guidance, method, agree = regex_g, "regex", None

    risk_count = len(_RISK_HEADING.findall(risk))
    senti = sentiment.score_text(mda)

    return SignalSet(
        doc_id=doc.doc_id,
        revenue_guidance=guidance["revenue_guidance"],
        capex_guidance=guidance["capex_guidance"],
        risk_factor_count=risk_count,
        sentiment_score=round(senti.score, 4),
        sentiment_backend=senti.backend,
        method=method,
        xcheck_agree=agree,
    )


def _section_text(doc: Document, items: tuple[str, ...]) -> str:
    txt = " ".join(s.text for s in doc.sections if s.item in items)
    return txt or doc.full_text
