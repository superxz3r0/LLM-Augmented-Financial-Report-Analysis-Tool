from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .ingest import Document

_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class DiffItem:
    kind: str
    item: str
    similarity: float
    old_text: str
    new_text: str


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if len(p.strip()) > 80]


def token_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class _Similarity:
    def __init__(self, old: list[str], new: list[str]):
        self.backend = "jaccard"
        self.old, self.new = old, new
        try:
            import numpy as np
            from .config import get_embedder
            model = get_embedder()
            self.e_old = model.encode(old, normalize_embeddings=True)
            self.e_new = model.encode(new, normalize_embeddings=True)
            self.np = np
            self.backend = "embeddings"
        except Exception:
            self._old_sets = [set(o.lower().split()) for o in old]

    def best_match(self, j: int) -> tuple[int, float]:
        if self.backend == "embeddings":
            sims = self.e_old @ self.e_new[j]
            i = int(sims.argmax())
            return i, float(sims[i])
        nset = set(self.new[j].lower().split())
        best_i, best = -1, 0.0
        for i, oset in enumerate(self._old_sets):
            if not oset or not nset:
                continue
            r = len(oset & nset) / len(oset | nset)
            if r > best:
                best_i, best = i, r
        return best_i, best


def diff_documents(old: Document, new: Document, max_items: int = 40) -> list[DiffItem]:
    results: list[DiffItem] = []
    old_by_item = {s.item: s for s in old.sections}

    for new_sec in new.sections:
        old_sec = old_by_item.get(new_sec.item)
        if old_sec is None:
            results.append(DiffItem("new", new_sec.item, 0.0, "", new_sec.text[:600]))
            continue

        old_paras, new_paras = _paragraphs(old_sec.text), _paragraphs(new_sec.text)
        if not new_paras:
            continue
        if not old_paras:
            old_paras = [""]

        sim = _Similarity(old_paras, new_paras)
        matched_old: set[int] = set()

        if sim.backend == "embeddings":
            t_boiler = settings.boilerplate_min_ratio
            t_subst = settings.substantive_similarity_threshold
            t_new = 0.45
        else:
            t_boiler, t_subst, t_new = 0.90, 0.65, 0.30

        for j, np_text in enumerate(new_paras):
            i, score = sim.best_match(j)
            if i >= 0:
                matched_old.add(i)
            if score >= t_boiler:
                continue
            kind = ("new" if score < t_new
                    else "substantive" if score < t_subst
                    else "minor")
            # Embeddings can rate two paragraphs as similar purely on shared
            # risk-disclosure vocabulary (e.g. two unrelated "risk factor"
            # paragraphs) even when they share almost no actual content.
            # Lexical overlap catches that false match, so treat very low
            # word overlap with the matched paragraph as "new" regardless
            # of the embedding score.
            if sim.backend == "embeddings" and kind != "new" and i >= 0 \
                    and token_jaccard(np_text, old_paras[i]) < 0.15:
                kind = "new"
            results.append(DiffItem(kind, new_sec.item, round(score, 3),
                                    old_paras[i][:600] if i >= 0 else "", np_text[:600]))

        for i, op in enumerate(old_paras):
            if i not in matched_old and len(op) > 80:
                results.append(DiffItem("removed", new_sec.item, 0.0, op[:600], ""))

    order = {"new": 0, "substantive": 1, "removed": 2, "minor": 3}
    results.sort(key=lambda d: order[d.kind])
    return results[:max_items]
