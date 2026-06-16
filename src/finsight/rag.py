from __future__ import annotations

import os
from dataclasses import dataclass

from .config import settings

SYSTEM_PROMPT = """You are a financial filings analyst. Answer the question using ONLY the numbered context passages provided. Cite passages inline as [1], [2] etc. after each claim. If the context does not contain the answer, say so explicitly — do not guess. Keep answers under 200 words."""


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


def _dedupe(hits: list, k: int) -> list:
    from .diff import token_jaccard
    kept: list = []
    for chunk, score in hits:
        if any(token_jaccard(chunk.text, kc.text) > 0.85 for kc, _ in kept):
            continue
        kept.append((chunk, score))
        if len(kept) >= k:
            break
    return kept


def answer(index, question: str, ticker: str | None = None, k: int | None = None,
           history: list[tuple[str, str]] | None = None) -> RagAnswer:
    k = k or settings.top_k
    hits = _dedupe(index.search(question, k * 3, ticker=ticker), k)
    if not hits:
        return RagAnswer("No relevant passages found in the corpus.", [], "none")

    context = _format_context(hits)
    if history:
        recent = history[-2:]
        convo = "\n".join(f"Q: {q}\nA: {a[:400]}" for q, a in recent)
        question = f"(Recent conversation for context:\n{convo})\n\nCurrent question: {question}"

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return RagAnswer(_openai_generate(question, context), hits, "openai")
        except Exception as e:
            print(f"[rag] OpenAI failed: {e}")
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return RagAnswer(_gemini_generate(question, context), hits, "gemini")
        except Exception as e:
            print(f"[rag] Gemini failed: {e}")

    lines = ["(Extractive mode — set OPENAI_API_KEY or GEMINI_API_KEY for generated answers.)\n"]
    for i, (c, score) in enumerate(hits, 1):
        snippet = c.text[:350].rsplit(" ", 1)[0]
        lines.append(f"[{i}] {snippet}…")
    return RagAnswer("\n\n".join(lines), hits, "extractive")
