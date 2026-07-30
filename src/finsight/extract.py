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
_RISK_CUE = re.compile(
    r"\b(?:risk\w*|advers\w*|could|may|uncertain\w*|depend\w*|rely|reliance|"
    r"harm\w*|expos\w*|fail\w*|litigation|subject to|ability|competition|"
    r"impact\w*|affect\w*|challeng\w*|vulnerab\w*|threat\w*|disrupt\w*|"
    r"fluctuat\w*|declin\w*|loss\w*|unable|inability)\b",
    re.IGNORECASE,
)
_INLINE_RISK_PREFIX = re.compile(r"^([^.!?]{3,40}\.)\s+\S")
_GLUED_RISK_BOUNDARY = re.compile(
    r"(?<=[a-z\u2019)])\s+(?=(?:The|Our|We|There|Changes|Legislation|Increasing|In|"
    r"[A-Z][a-z]{3,}(?:[\u2019']s)?)\b)"
)
_TITLE_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "our", "the", "to", "with", "we", "us", "is", "are", "may",
    "could", "not",
}


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
    text = mda[:12000]
    try:
        if os.environ.get("GEMINI_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel(settings_model(), system_instruction=_LLM_PROMPT)
            return _validate_llm_payload(model.generate_content(text).text)
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            from .config import settings as _s
            resp = OpenAI().chat.completions.create(
                model=_s.openai_model, temperature=0,
                messages=[{"role": "system", "content": _LLM_PROMPT},
                          {"role": "user", "content": text}])
            return _validate_llm_payload(resp.choices[0].message.content)
    except Exception:
        return None
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


def _count_risk_factors(text: str) -> int:
    """Count risk-factor headings from paragraph structure.

    SEC text conversions commonly preserve bold headings in one of three
    forms: a separate paragraph, a short ``Heading.`` prefix, or a heading
    glued directly to its explanatory paragraph.  Counting those structures
    is substantially less sensitive to arbitrary line wrapping than the
    original one-line keyword regex, which remains as a fallback.
    """
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    if not paragraphs:
        return 0

    def is_noise(p: str) -> bool:
        lower = p.lower()
        return (
            "form 10-k" in lower
            or "table of contents" in lower
            or lower.startswith(("item 1b", "risk factor summary"))
            or bool(re.fullmatch(r"[\d\s|]+", p))
        )

    # Most filings retain each bold risk heading as a short paragraph followed
    # by a materially longer explanatory paragraph. Bulleted risk statements
    # remain eligible because some filings use bullets as the heading itself.
    structural_count = sum(
        40 <= len(p) <= 280
        and not is_noise(p)
        and bool(_RISK_CUE.search(p))
        and i + 1 < len(paragraphs)
        and len(paragraphs[i + 1]) >= 250
        for i, p in enumerate(paragraphs)
    )

    # Some filers use consistently title-cased headings without explicit risk
    # words.  Only use this mode when it is clearly the document-wide template.
    def is_title_case(p: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z\u2019'-]*", p)
        content = [w for w in words if w.lower() not in _TITLE_STOPWORDS]
        return (
            len(content) >= 2
            and sum(w[0].isupper() for w in content) / len(content) >= 0.8
        )

    title_count = sum(
        20 <= len(p) <= 160
        and not is_noise(p)
        and is_title_case(p)
        and i + 1 < len(paragraphs)
        and len(paragraphs[i + 1]) >= 150
        for i, p in enumerate(paragraphs)
    )
    if title_count >= 15:
        return title_count

    # Compact layouts often encode headings inside the same paragraph.  A
    # dominant short "Heading." prefix or a glued heading/body boundary is a
    # stronger signal for these documents than paragraph length alone.
    if len(paragraphs) < 60:
        inline_count = 0
        for p in paragraphs:
            match = _INLINE_RISK_PREFIX.match(p)
            if not match:
                continue
            prefix = match.group(1)
            if (
                len(prefix[:-1].split()) <= 5
                and not prefix.lower().startswith(("item ", "the oil, gas"))
            ):
                inline_count += 1
        if inline_count >= 10:
            return inline_count

        glued_count = 0
        for p in paragraphs:
            for match in _GLUED_RISK_BOUNDARY.finditer(p[:165]):
                prefix = p[:match.start()]
                if len(prefix) >= 30 and _RISK_CUE.search(prefix):
                    glued_count += 1
                    break
        if glued_count >= 8:
            return glued_count

    line_count = len(_RISK_HEADING.findall(text))
    # Repeated page headers fragment paragraphs.  Blending the conservative
    # line count with the structural count prevents either representation from
    # dominating this known conversion artifact.
    if len(re.findall(r"table of contents", text, re.IGNORECASE)) >= 8 \
            and structural_count:
        return round((structural_count + line_count) / 2)
    return structural_count or line_count


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

    risk_count = _count_risk_factors(risk)
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
