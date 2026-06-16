from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[\d+\]")
_NUM = re.compile(r"\$?\d[\d,.]*\s*(?:billion|million|bn|m\b|percent|%|basis points)?",
                  re.IGNORECASE)
_WORD = re.compile(r"[a-z]{3,}")
_STOP = {"the", "and", "for", "that", "this", "with", "are", "was", "were", "has",
         "have", "its", "their", "from", "which", "also", "may", "could", "would"}

EMB_THRESHOLD = 0.62
JACCARD_THRESHOLD = 0.18
PASS_RATIO = 0.80


@dataclass
class SentenceAudit:
    sentence: str
    support: float
    grounded: bool
    unsupported_numbers: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    passed: bool
    grounded_ratio: float
    backend: str
    sentences: list[SentenceAudit]

    def summary(self) -> str:
        n = len(self.sentences)
        g = sum(s.grounded for s in self.sentences)
        verdict = "PASSED" if self.passed else "FAILED"
        return f"Hallucination audit {verdict}: {g}/{n} sentences grounded ({self.backend})"


def _normalise_number(s: str) -> str:
    return re.sub(r"[\s,$]", "", s.lower()).replace("percent", "%")


def _content_tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _is_meta(sentence: str) -> bool:
    s = sentence.lower()
    return ("context does not" in s or "extractive mode" in s
            or "no relevant passages" in s or len(_content_tokens(sentence)) < 4)


def audit_answer(answer: str, source_chunks: list) -> AuditReport:
    sources = [c.text for c in source_chunks]
    if not sources:
        return AuditReport(False, 0.0, "none", [])

    clean = _CITE.sub("", answer)
    sentences = [s.strip() for s in _SENT_SPLIT.split(clean)
                 if s.strip() and not _is_meta(s)]
    if not sentences:
        return AuditReport(True, 1.0, "trivial", [])

    backend = "jaccard"
    scores: list[float] = []
    try:
        from sentence_transformers import SentenceTransformer
        from .config import settings
        model = SentenceTransformer(settings.embedding_model)
        e_sent = model.encode(sentences, normalize_embeddings=True)
        e_src = model.encode(sources, normalize_embeddings=True)
        sims = e_sent @ e_src.T
        scores = sims.max(axis=1).tolist()
        backend, threshold = "embeddings", EMB_THRESHOLD
    except Exception:
        src_tokens = [_content_tokens(s) for s in sources]
        for sent in sentences:
            st = _content_tokens(sent)
            best = max((len(st & src) / max(len(st | src), 1) for src in src_tokens),
                       default=0.0)
            scores.append(best)
        threshold = JACCARD_THRESHOLD

    all_source_numbers = {_normalise_number(n) for s in sources for n in _NUM.findall(s)}
    audits: list[SentenceAudit] = []
    for sent, score in zip(sentences, scores):
        nums = [n.strip() for n in _NUM.findall(sent) if any(ch.isdigit() for ch in n)]
        missing = [n for n in nums if _normalise_number(n) not in all_source_numbers]
        grounded = score >= threshold and not missing
        audits.append(SentenceAudit(sent, round(float(score), 3), grounded, missing))

    ratio = sum(a.grounded for a in audits) / len(audits)
    any_bad_numbers = any(a.unsupported_numbers for a in audits)
    return AuditReport(ratio >= PASS_RATIO and not any_bad_numbers,
                       round(ratio, 3), backend, audits)
