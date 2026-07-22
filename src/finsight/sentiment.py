from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .config import settings

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

DEFAULT_MAX_SENTENCES = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_INFERENCE_CHUNK_SIZE = 512


def _notify_progress(
    progress: Callable[[int, int], None] | None,
    done: int,
    total: int,
) -> None:
    """Keep presentation-layer callback failures out of model fallback logic."""
    if progress is None:
        return
    try:
        progress(done, total)
    except Exception as e:
        print(f"[sentiment] progress callback failed ({type(e).__name__})")

_LM_POSITIVE = {
    "achieve", "achieved", "advantage", "benefit", "beneficial", "best", "exceed",
    "exceeded", "gain", "gains", "growth", "improve", "improved", "improvement",
    "innovation", "opportunity", "opportunities", "outperform", "profitable",
    "record", "strength", "strengthen", "strong", "succeed", "success", "successful",
}
_LM_NEGATIVE = {
    "adverse", "adversely", "against", "challenge", "challenges", "concern",
    "concerns", "decline", "declined", "decrease", "decreased", "deficit",
    "difficult", "downturn", "failure", "impair", "impairment", "litigation",
    "loss", "losses", "negative", "risk", "risks", "uncertain", "uncertainty",
    "volatile", "volatility", "weak", "weakness", "writedown",
}


@dataclass
class SentimentResult:
    score: float
    n_sentences: int
    backend: str
    breakdown: dict


class _FinBert:
    _pipe = None

    @classmethod
    def pipe(cls):
        if cls._pipe is None:
            from transformers import pipeline
            cls._pipe = pipeline("text-classification", model=settings.finbert_model,
                                 top_k=None, truncation=True, max_length=512)
        return cls._pipe

    @classmethod
    def score(cls, sentences: list[str]) -> SentimentResult:
        return cls.score_many([sentences])[0]

    @classmethod
    def score_many(
        cls,
        sentence_groups: list[list[str]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        inference_chunk_size: int = DEFAULT_INFERENCE_CHUNK_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[SentimentResult]:
        """Score many documents while sharing full inference batches.

        Sentence ownership is retained while the inputs are flattened, so
        probabilities can be accumulated back into the original document.
        The outer inference chunks bound peak memory; ``batch_size`` is passed
        to the transformers pipeline inside each chunk.
        """
        if batch_size <= 0 or inference_chunk_size <= 0:
            raise ValueError("batch_size and inference_chunk_size must be positive")

        owners: list[int] = []
        flat_sentences: list[str] = []
        for doc_idx, sentences in enumerate(sentence_groups):
            owners.extend([doc_idx] * len(sentences))
            flat_sentences.extend(sentences)

        totals = [[0.0, 0.0, 0.0] for _ in sentence_groups]
        total_sentences = len(flat_sentences)
        _notify_progress(progress, 0, total_sentences)

        pipe = cls.pipe() if total_sentences else None
        for start in range(0, total_sentences, inference_chunk_size):
            stop = min(start + inference_chunk_size, total_sentences)
            outputs = pipe(flat_sentences[start:stop], batch_size=batch_size)
            if len(outputs) != stop - start:
                raise RuntimeError("FinBERT returned a different number of results than inputs")
            for owner, result in zip(owners[start:stop], outputs):
                probs = {d["label"].lower(): d["score"] for d in result}
                totals[owner][0] += probs.get("positive", 0.0)
                totals[owner][1] += probs.get("negative", 0.0)
                totals[owner][2] += probs.get("neutral", 0.0)
            _notify_progress(progress, stop, total_sentences)

        scored = []
        for sentences, (pos, neg, neu) in zip(sentence_groups, totals):
            n = len(sentences)
            if n == 0:
                scored.append(SentimentResult(0.0, 0, "none", {}))
                continue
            scored.append(SentimentResult(
                score=(pos - neg) / n,
                n_sentences=n,
                backend="finbert",
                breakdown={"positive": pos / n, "negative": neg / n, "neutral": neu / n},
            ))
        return scored


def _lexicon_score(sentences: list[str]) -> SentimentResult:
    pos = neg = 0
    total_tokens = 0
    for s in sentences:
        tokens = re.findall(r"[a-z]+", s.lower())
        total_tokens += len(tokens)
        pos += sum(t in _LM_POSITIVE for t in tokens)
        neg += sum(t in _LM_NEGATIVE for t in tokens)
    denom = max(pos + neg, 1)
    return SentimentResult(
        score=(pos - neg) / denom, n_sentences=len(sentences), backend="lexicon",
        breakdown={"positive": pos / max(total_tokens, 1),
                   "negative": neg / max(total_tokens, 1),
                   "neutral": 1.0 - (pos + neg) / max(total_tokens, 1)},
    )


def _sentences(text: str, max_sentences: int) -> list[str]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]
    return sentences[:max_sentences]


def score_texts(
    texts: list[str],
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    inference_chunk_size: int = DEFAULT_INFERENCE_CHUNK_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> list[SentimentResult]:
    """Score multiple texts in shared FinBERT batches.

    If FinBERT cannot be loaded or inference fails, each document falls back
    to the existing deterministic financial lexicon scorer.
    """
    sentence_groups = [_sentences(text, max_sentences) for text in texts]
    if not any(sentence_groups):
        _notify_progress(progress, 0, 0)
        return [SentimentResult(0.0, 0, "none", {}) for _ in texts]
    try:
        return _FinBert.score_many(
            sentence_groups,
            batch_size=batch_size,
            inference_chunk_size=inference_chunk_size,
            progress=progress,
        )
    except Exception as e:
        print(f"[sentiment] FinBERT unavailable ({type(e).__name__}); using lexicon fallback")
        total = sum(len(sentences) for sentences in sentence_groups)
        _notify_progress(progress, total, total)
        return [
            _lexicon_score(sentences) if sentences else SentimentResult(0.0, 0, "none", {})
            for sentences in sentence_groups
        ]


def score_text(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> SentimentResult:
    """Backward-compatible single-document sentiment API."""
    return score_texts([text], max_sentences=max_sentences)[0]
