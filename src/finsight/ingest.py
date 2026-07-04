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


def _split_sections(text: str) -> list[Section]:
    """Split filing text on 'Item N.' headings; fall back to one big section.

    TOC handling: every heading occurrence becomes a candidate; candidates
    whose body is shorter than _MIN_SECTION_CHARS are dropped, and when the
    same Item number appears more than once (TOC + body + cross-references)
    the occurrence with the longest body wins.
    """
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return [Section(item="0", title="Full document", text=text.strip())]

    best: dict[str, Section] = {}
    order: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < _MIN_SECTION_CHARS:
            continue
        item = m.group(1).upper()
        if item not in best or len(body) > len(best[item].text):
            if item not in best:
                order.append(item)
            best[item] = Section(item=item, title=m.group(2).strip(" .:\u2013\u2014-"), text=body)

    sections = [best[i] for i in order]
    return sections or [Section(item="0", title="Full document", text=text.strip())]


# --- earnings-call transcripts -------------------------------------------
# Transcripts have no "Item N" structure; their structure is speaker turns.
# We recognise the Motley Fool / Seeking Alpha header style:
#     Tim Cook -- Chief Executive Officer
#     Erik Woodring -- Morgan Stanley -- Analyst
#     Operator
# and the Prepared-Remarks vs Q&A boundary, so every chunk can later cite
# WHO said it and in WHICH phase of the call.
_SPEAKER_LINE = re.compile(
    r"^([A-Z][\w.'\u2019-]*(?:\s+[A-Z][\w.'\u2019-]*){0,3})"   # 1-4 capitalised name words
    r"\s*(?:--|\u2014|\u2013)\s*"                              # -- or em/en dash
    r"([A-Z].{1,80})$"                                          # role/title, starts uppercase
)
_OPERATOR_LINE = re.compile(r"^Operator\s*:?\s*$", re.IGNORECASE)
_QA_BOUNDARY = re.compile(
    r"^\s*(?:question[s]?[\s\-]*(?:&|and)?[\s\-]*answer|q\s*&\s*a\b)", re.IGNORECASE)
# API-sourced transcripts (e.g. API Ninjas) use "Name: speech..." on one line
# instead of a standalone "Name -- Title" header. Require 2-4 capitalised
# words (or Operator) so prose like "Note:" / "Contents:" doesn't match.
_SPEAKER_COLON = re.compile(
    r"^((?:[A-Z][\w.'\u2019-]+\s+){1,3}[A-Z][\w.'\u2019-]+|Operator)\s*:\s+(\S.*)$")
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
    splitter = _split_transcript if form.upper() == "TRANSCRIPT" else _split_sections
    return Document(ticker=ticker, form=form, date=date, path=path,
                    sections=splitter(text))


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