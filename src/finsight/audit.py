from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
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

    # Split (and record each sentence's cited chunk numbers) before stripping
    # the [n] markers, so grounding can be checked against the *specific*
    # chunk(s) a sentence claims to cite rather than the whole retrieved pool
    # — a sentence citing [1] but whose number only appears in [4] should
    # fail, even though that number is present "somewhere" in the sources.
    # Handles grouped citations like "[1, 2]" as well as single "[1]".
    sentences: list[str] = []
    cited_sets: list[set[int]] = []
    for raw in _SENT_SPLIT.split(answer):
        cited = {int(n) for group in _CITE.findall(raw) for n in re.findall(r"\d+", group)}
        clean_s = _CITE.sub("", raw).strip()
        if clean_s and not _is_meta(clean_s):
            sentences.append(clean_s)
            cited_sets.append(cited)
    if not sentences:
        return AuditReport(True, 1.0, "trivial", [])

    def _candidate_idxs(cited: set[int]) -> list[int]:
        valid = [i - 1 for i in cited if 1 <= i <= len(sources)]
        return valid or list(range(len(sources)))

    backend = "jaccard"
    scores: list[float] = []
    try:
        from .config import get_embedder
        model = get_embedder()
        e_sent = model.encode(sentences, normalize_embeddings=True)
        e_src = model.encode(sources, normalize_embeddings=True)
        sims = e_sent @ e_src.T
        for row, cited in zip(sims, cited_sets):
            idxs = _candidate_idxs(cited)
            scores.append(float(max(row[i] for i in idxs)))
        backend, threshold = "embeddings", EMB_THRESHOLD
    except Exception:
        src_tokens = [_content_tokens(s) for s in sources]
        for sent, cited in zip(sentences, cited_sets):
            st = _content_tokens(sent)
            idxs = _candidate_idxs(cited)
            best = max((len(st & src_tokens[i]) / max(len(st | src_tokens[i]), 1) for i in idxs),
                       default=0.0)
            scores.append(best)
        threshold = JACCARD_THRESHOLD

    all_source_numbers = {_normalise_number(n) for s in sources for n in _NUM.findall(s)}
    audits: list[SentenceAudit] = []
    for sent, score, cited in zip(sentences, scores, cited_sets):
        nums = [n.strip() for n in _NUM.findall(sent) if any(ch.isdigit() for ch in n)]
        if cited:
            idxs = _candidate_idxs(cited)
            allowed_numbers = {_normalise_number(n) for i in idxs for n in _NUM.findall(sources[i])}
        else:
            allowed_numbers = all_source_numbers
        missing = [n for n in nums if _normalise_number(n) not in allowed_numbers]
        grounded = score >= threshold and not missing
        audits.append(SentenceAudit(sent, round(float(score), 3), grounded, missing))

    ratio = sum(a.grounded for a in audits) / len(audits)
    any_bad_numbers = any(a.unsupported_numbers for a in audits)
    return AuditReport(ratio >= PASS_RATIO and not any_bad_numbers,
                       round(ratio, 3), backend, audits)
