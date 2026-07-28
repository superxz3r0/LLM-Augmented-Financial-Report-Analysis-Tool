"""Ingestion: turn raw filing files into structured Document objects.

Real filings are fetched with scripts/fetch_filings.py (needs internet).
This module is deliberately source-agnostic: it only cares about files
on disk that follow the naming convention

    <TICKER>_<FORM>_<YYYY-MM-DD>.txt      e.g.  AAPL_10-K_2025-11-01.txt

so the same code path serves the bundled sample data and real EDGAR data.

Real-EDGAR hardening:
  * HTML is detected and stripped (EDGAR full-text is frequently HTML).
  * Whitespace/entity noise is normalised before section splitting.
  * Table-of-contents "Item N" lines are skipped by requiring a minimum
    body length per section, and duplicate Item numbers keep the longest
    body (filings often mention "Item 1A" in cross-references).
"""
from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Section headers we try to split 10-K/10-Q text on. EDGAR filings keep the
# "Item N." headings, which is enough structure for retrieval and for the
# diff engine to align sections across years.
ITEM_PATTERN = re.compile(
    r"^\s*(?:ITEM|Item|I\s?T\s?E\s?M)\s+(\d{1,2}A?B?)\s*[.:\u2013\u2014-]?\s*(.{0,120})$",
    re.MULTILINE,
)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_MIN_SECTION_CHARS = 200          # below this it's a TOC entry / cross-reference

# Fallback for filers (JPMorgan's real 10-K/10-Q are confirmed cases) whose
# body text states each "Item N." heading only once, in the table of
# contents, and never repeats it before the section itself — the section
# instead starts with a bare descriptive heading and no item number at all.
# Against that structure ITEM_PATTERN only ever matches inside the TOC
# block, so the *last* TOC entry's unbounded "body" swallows the entire rest
# of the document under one item number (see _split_sections). These are
# standard SEC item titles used verbatim as bare standalone headings by at
# least some filers; matching them lets that content be recovered under its
# real item number instead of being lost inside whichever item happened to
# be last in the TOC. This is inherently best-effort — filers are free to
# caption sections however they like, and not every item is guaranteed to be
# recoverable this way for every filer.
#
# 10-K and 10-Q item *numbers* diverge for the same content (MD&A is Item 7
# in a 10-K but Item 2 in a 10-Q), so each heading phrase maps to a
# form-specific item number. 10-Q Part II reuses Item numbers 1-6 for
# unrelated topics (Legal Proceedings, Risk Factors, etc., distinct from
# Part I's Items 1-4) — too ambiguous to resolve from the item number alone,
# so only Part I's essentially unambiguous items are mapped for 10-Qs.
_BARE_HEADING_PATTERNS: dict[str, str] = {
    "risk_factors": r"risk factors",
    "legal_proceedings": r"legal proceedings",
    "mdna": r"management[’']s discussion and analysis",
    "market_risk": r"quantitative and qualitative disclosures about market risk",
    "fin_statements": r"financial statements and supplementary data",
    "controls": r"controls and procedures",
}
_BARE_HEADING_RES = {key: re.compile(rf"^\s*({pat})\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
                     for key, pat in _BARE_HEADING_PATTERNS.items()}
_BARE_ITEM_10K: dict[str, str] = {
    "risk_factors": "1A", "legal_proceedings": "3", "mdna": "7",
    "market_risk": "7A", "fin_statements": "8", "controls": "9A",
}
_BARE_ITEM_10Q: dict[str, str] = {"mdna": "2", "market_risk": "3", "controls": "4"}


@dataclass
class Section:
    item: str           # e.g. "1A"
    title: str          # e.g. "Risk Factors"
    text: str


@dataclass
class Document:
    ticker: str
    form: str           # "10-K", "10-Q", "TRANSCRIPT"
    date: str           # ISO date string
    path: Path
    sections: list[Section] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.form}_{self.date}"

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def clean_text(raw: str) -> str:
    """Strip HTML if present and normalise whitespace, preserving paragraph
    breaks (the diff engine depends on blank-line paragraph boundaries)."""
    text = raw
    if "</" in text[:5000] or "<html" in text[:2000].lower():
        text = _SCRIPT_STYLE.sub(" ", text)
        # block-level tags become paragraph breaks so structure survives
        text = re.sub(r"</?(p|div|tr|table|h\d|li|br)[^>]*>", "\n\n", text, flags=re.IGNORECASE)
        text = _TAG.sub(" ", text)
        text = html_lib.unescape(text)
    text = text.replace("\u00a0", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sections_from_spans(text: str, spans: list[tuple[str, str, int]]) -> list[Section]:
    """Shared TOC-vs-body resolution: spans is (item, title, body_start) in
    document order; each body runs to the next span's start (or EOF).
    Candidates whose body is shorter than _MIN_SECTION_CHARS are dropped
    (TOC entries / cross-references), and when the same item number appears
    more than once the occurrence with the longest body wins."""
    best: dict[str, Section] = {}
    order: list[str] = []
    for i, (item, title, start) in enumerate(spans):
        end = spans[i + 1][2] if i + 1 < len(spans) else len(text)
        body = text[start:end].strip()
        if len(body) < _MIN_SECTION_CHARS:
            continue
        if item not in best or len(body) > len(best[item].text):
            if item not in best:
                order.append(item)
            best[item] = Section(item=item, title=title, text=body)
    return [best[i] for i in order]


def _split_sections(text: str, form: str = "10-K") -> list[Section]:
    """Split filing text on 'Item N.' headings, augmented with a fallback
    for bare (unprefixed) versions of those same headings \u2014 see
    _BARE_HEADING_PATTERNS. Both heading styles are merged into one span
    list, in document order, and resolved together by _sections_from_spans:
    whichever style actually located a given item's real content ends up
    winning that item's longest-body slot, and a bare-heading span
    downstream of a TOC-only "Item N." span correctly caps that item's
    runaway body instead of letting it swallow the rest of the document.
    Items no filer wrote a heading style this recognises for just don't get
    their own section, same as always.
    """
    matches = list(ITEM_PATTERN.finditer(text))
    item_spans = [(m.group(1).upper(), m.group(2).strip(" .:\u2013\u2014-"), m.end()) for m in matches]

    bare_item = _BARE_ITEM_10Q if form.upper() == "10-Q" else _BARE_ITEM_10K
    bare_spans = [(bare_item[key], m.group(1).strip(), m.end())
                  for key in bare_item for m in _BARE_HEADING_RES[key].finditer(text)]
    spans = sorted(item_spans + bare_spans, key=lambda s: s[2])

    sections = _sections_from_spans(text, spans)
    return sections or [Section(item="0", title="Full document", text=text.strip())]


# --- earnings-call transcripts -------------------------------------------
# Transcripts have no "Item N" structure; their structure is speaker turns.
# We recognise the Motley Fool / Seeking Alpha header style:
#     Tim Cook -- Chief Executive Officer
#     Erik Woodring -- Morgan Stanley -- Analyst
#     Operator
# and the Prepared-Remarks vs Q&A boundary, so every chunk can later cite
# WHO said it and in WHICH phase of the call.

#don't need every first letter is uppercase letter
_NAME = r"[A-Z][\w.'\u2019-]*"
_PART = r"(?:de|del|della|der|den|di|da|dos|du|la|le|van|von|ter|ten|te|bin|al|el)"

_SPEAKER_LINE = re.compile(
    r"^(" + _NAME + r"(?:\s+(?:" + _PART + r"\s+){0,2}" + _NAME + r"){0,3})"
    r"\s*(?:--|\u2014|\u2013)\s*"
    r"([A-Z].{1,80})$"
)

_OPERATOR_LINE = re.compile(r"^Operator\s*:?\s*$", re.IGNORECASE)
_QA_BOUNDARY = re.compile(
    r"^\s*(?:question[s]?[\s\-]*(?:&|and)?[\s\-]*answer|q\s*&\s*a\b)", re.IGNORECASE)
# API-sourced transcripts (e.g. API Ninjas) use "Name: speech..." on one line
# instead of a standalone "Name -- Title" header. Require 2-4 capitalised
# words (or Operator) so prose like "Note:" / "Contents:" doesn't match.

_SPEAKER_COLON = re.compile(
    r"^(" + _NAME + r"(?:\s+(?:" + _PART + r"\s+){0,2}" + _NAME + r"){1,3}|Operator)"
    r"\s*:\s+(\S.*)$")

# Those same API transcripts often lack an explicit "Questions and Answers"
# header line, so also flip to Q&A when the operator hands over to questions.
_QA_HEURISTIC = re.compile(
    r"(?:first question|now begin.{0,30}question|open the (?:line|floor|call).{0,30}question)",
    re.IGNORECASE)
_MIN_TURN_CHARS = 30      # drop "Thank you." style micro-turns / noise


def _split_transcript(text: str) -> list[Section]:
    """Split an earnings-call transcript on speaker turns.

    Section.item  -> "PR" (prepared remarks) or "QA"
    Section.title -> "Speaker Name (Role)"
    Section.text  -> that speaker's turn

    Chunks produced downstream therefore never mix two speakers, and the
    citation can say e.g.  AURB Earnings Call (2025-12-20), Q&A — Jane Doe (CFO).
    Falls back to a single full-document section if no speaker structure found.
    """
    phase = "PR"
    speaker, role = "", ""
    buf: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        nonlocal buf
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(buf)).strip()
        if speaker and len(body) >= _MIN_TURN_CHARS:
            title = f"{speaker} ({role})" if role else speaker
            sections.append(Section(item=phase, title=title, text=body))
        buf = []

    for raw in text.split("\n"):
        line = raw.strip()
        if len(line) < 90:                          # speaker/boundary lines are short
            if _QA_BOUNDARY.match(line) and len(line) < 60:
                flush()
                phase = "QA"
                continue
            m = _SPEAKER_LINE.match(line)
            if m:
                flush()
                speaker, role = m.group(1).strip(), m.group(2).strip()
                continue
            if _OPERATOR_LINE.match(line):
                flush()
                speaker, role = "Operator", ""
                continue
        m2 = _SPEAKER_COLON.match(line)             # "Name: speech..." (API style)
        if m2:
            flush()
            speaker, role = m2.group(1).strip(), ""
            buf.append(m2.group(2))
        else:
            buf.append(raw)
        if phase == "PR" and speaker == "Operator" and _QA_HEURISTIC.search(line):
            phase = "QA"                            # operator hands over to questions
    flush()

    return sections or [Section(item="0", title="Full transcript", text=text.strip())]


def load_document(path: Path) -> Document:
    name = path.stem                       # AAPL_10-K_2025-11-01
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad filename (want TICKER_FORM_DATE.txt): {path.name}")
    ticker, form, date = parts
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if form.upper() == "TRANSCRIPT":
        sections = _split_transcript(text)
    else:
        sections = _split_sections(text, form)
    return Document(ticker=ticker, form=form, date=date, path=path, sections=sections)


def load_corpus(*dirs: Path) -> list[Document]:
    """Load every .txt/.htm/.html filing found in the given directories."""
    docs: list[Document] = []
    for d in dirs:
        if not d.exists():
            continue
        for pattern in ("*.txt", "*.htm", "*.html"):
            for p in sorted(d.glob(pattern)):
                try:
                    docs.append(load_document(p))
                except ValueError:
                    continue
    return docs