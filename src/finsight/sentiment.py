from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

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
        results = cls.pipe()(sentences, batch_size=16)
        pos = neg = neu = 0.0
        for r in results:
            probs = {d["label"].lower(): d["score"] for d in r}
            pos += probs.get("positive", 0.0)
            neg += probs.get("negative", 0.0)
            neu += probs.get("neutral", 0.0)
        n = max(len(sentences), 1)
        return SentimentResult(
            score=(pos - neg) / n, n_sentences=n, backend="finbert",
            breakdown={"positive": pos / n, "negative": neg / n, "neutral": neu / n},
        )


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


def score_text(text: str, max_sentences: int = 200) -> SentimentResult:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]
    sentences = sentences[:max_sentences]
    if not sentences:
        return SentimentResult(0.0, 0, "none", {})
    try:
        return _FinBert.score(sentences)
    except Exception as e:
        print(f"[sentiment] FinBERT unavailable ({type(e).__name__}); using lexicon fallback")
        return _lexicon_score(sentences)
