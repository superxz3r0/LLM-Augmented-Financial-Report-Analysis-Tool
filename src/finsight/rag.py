from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .config import settings

_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _group_numbers(group: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", group)]

SYSTEM_PROMPT = """You are a financial filings analyst. Answer the question using ONLY the numbered context passages provided. Cite passages inline as [1], [2] etc. after each claim. If the context does not contain the answer, say so explicitly — do not guess.

Some passages contain financial tables that have been flattened into plain text, so column headers (e.g. fiscal years or segment names) may sit on their own line above rows of unlabeled numbers. Before citing a figure, re-check which column and row it actually belongs to; if that mapping is ambiguous, say so instead of guessing. Never compute, round, or infer a number that is not itself stated verbatim in the passages.

Keep answers under 200 words."""


@dataclass
class RagAnswer:
    text: str
    sources: list
    backend: str


def _format_context(hits) -> str:
    return "\n\n".join(f"[{i+1}] ({c.citation})\n{c.text}" for i, (c, _) in enumerate(hits))


def _openai_generate(question: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _gemini_generate(question: str, context: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=SYSTEM_PROMPT)
    resp = model.generate_content(f"Context:\n{context}\n\nQuestion: {question}")
    return resp.text


def _token_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _rerank(question: str, hits: list) -> list:
    """Cross-encoder re-scoring of the hybrid retrieval pool.

    RRF fusion (see index.py) is rank-based and fuses two independently
    computed rankings — it has no notion of how relevant a passage actually
    is to *this* question, just where it landed in each list. A
    cross-encoder reads the (question, passage) pair jointly and scores
    relevance directly, which is why it reliably outperforms bi-encoder /
    lexical fusion alone as a final ordering step. Falls back to the
    untouched hybrid order if the model can't load (e.g. first run with no
    internet to fetch it from Hugging Face).
    """
    if not hits:
        return hits
    try:
        import numpy as np

        from .config import get_reranker
        model = get_reranker()
        logits = model.predict([(question, c.text) for c, _ in hits])
        # the cross-encoder's raw logits (e.g. -11..+8) aren't meaningful to
        # show a user as "relevance" the way the old RRF score was at least
        # bounded — squash to [0, 1] so the sources panel reads sensibly.
        scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        return [(hits[i][0], float(scores[i])) for i in order]
    except Exception as e:
        print(f"[rag] reranker unavailable ({type(e).__name__}: {e}); using hybrid ranking order")
        return hits


def _dedupe(hits: list, k: int) -> list:
    kept: list = []
    for chunk, score in hits:
        if any(_token_jaccard(chunk.text, kc.text) > 0.85 for kc, _ in kept):
            continue
        kept.append((chunk, score))
        if len(kept) >= k:
            break
    return kept


def _drop_uncited(text: str, hits: list) -> tuple[str, list]:
    """Keep only the sources the answer actually cites, renumbering both the
    [n] markers in the text and the returned source list so they stay in
    lockstep (e.g. an answer citing only [1] and [3] out of 5 retrieved
    passages becomes [1] and [2], with sources trimmed to match) — instead of
    always showing every retrieved passage regardless of whether the answer
    used it. Handles grouped citations like "[1, 2]" (some models cite that
    way instead of "[1][2]"), not just single-number brackets. Falls back to
    the full set if the answer cites nothing (e.g. the extractive dump, which
    cites every retrieved chunk by construction, or a model that forgot to
    cite)."""
    all_cited = (n for m in _CITE_GROUP.finditer(text) for n in _group_numbers(m.group(1)))
    cited = sorted({n for n in all_cited if 1 <= n <= len(hits)})
    if not cited:
        return text, hits
    remap = {old: new for new, old in enumerate(cited, 1)}

    def _renumber(m):
        kept = [remap[n] for n in _group_numbers(m.group(1)) if n in remap]
        return f"[{', '.join(str(n) for n in kept)}]" if kept else m.group(0)

    renumbered = _CITE_GROUP.sub(_renumber, text)
    return renumbered, [hits[i - 1] for i in cited]


def answer(index, question: str, ticker: str | None = None, k: int | None = None,
           history: list[tuple[str, str]] | None = None) -> RagAnswer:
    k = k or settings.top_k
    pool = index.search(question, max(settings.rerank_pool, k * 3), ticker=ticker)
    hits = _dedupe(_rerank(question, pool), k)
    if not hits:
        return RagAnswer("No relevant passages found in the corpus.", [], "none")

    context = _format_context(hits)
    if history:
        recent = history[-2:]
        convo = "\n".join(f"Q: {q}\nA: {a[:400]}" for q, a in recent)
        question = f"(Recent conversation for context:\n{convo})\n\nCurrent question: {question}"

    text, backend = None, None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            text, backend = _openai_generate(question, context), "openai"
        except Exception as e:
            print(f"[rag] OpenAI failed: {e}")
    if text is None and os.environ.get("GEMINI_API_KEY"):
        try:
            text, backend = _gemini_generate(question, context), "gemini"
        except Exception as e:
            print(f"[rag] Gemini failed: {e}")

    if text is None:
        lines = ["(Extractive mode — set OPENAI_API_KEY or GEMINI_API_KEY for generated answers.)\n"]
        for i, (c, score) in enumerate(hits, 1):
            snippet = c.text[:350].rsplit(" ", 1)[0]
            lines.append(f"[{i}] {snippet}…")
        text, backend = "\n\n".join(lines), "extractive"

    text, used_hits = _drop_uncited(text, hits)
    return RagAnswer(text, used_hits, backend)
