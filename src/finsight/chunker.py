"""Chunking: split sections into overlapping passages for embedding.

We chunk on sentence boundaries where possible so retrieved passages read
naturally and citations point at coherent text, not mid-sentence fragments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .ingest import Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u201c\"(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    form: str
    date: str
    item: str
    section_title: str
    text: str

    @property
    def citation(self) -> str:
        if self.form.upper() == "TRANSCRIPT":
            phase = {"QA": "Q&A", "PR": "Prepared Remarks"}.get(self.item)
            head = f"{self.ticker} Earnings Call ({self.date})"
            if phase:
                return f"{head}, {phase} — {self.section_title}"
            return f"{head} — {self.section_title}"   # flat text: no speaker structure
        return f"{self.ticker} {self.form} ({self.date}), Item {self.item} — {self.section_title}"


def _pack_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Pack sentences into ~size-char chunks; overlap carries whole trailing
    sentences (never mid-word slices) so every chunk starts cleanly."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > size and cur:
            chunks.append(" ".join(cur))
            # carry whole sentences from the tail until overlap budget is met
            carried: list[str] = []
            budget = 0
            for prev in reversed(cur):
                if budget + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                budget += len(prev) + 1
            cur, cur_len = carried, budget
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    out: list[Chunk] = []
    n = 0
    for sec in doc.sections:
        sentences = _SENT_SPLIT.split(sec.text)
        for piece in _pack_sentences(sentences, settings.chunk_size, settings.chunk_overlap):
            out.append(Chunk(
                chunk_id=f"{doc.doc_id}#{n}",
                doc_id=doc.doc_id,
                ticker=doc.ticker,
                form=doc.form,
                date=doc.date,
                item=sec.item,
                section_title=sec.title,
                text=piece,
            ))
            n += 1
    return out


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    return chunks