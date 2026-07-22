from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable

from .config import settings
from .ingest import Document


# Included in the Streamlit study key so a hot-reloaded scoring change cannot
# leave results from an older disclosure definition visible in the session.
DISCLOSURE_SIGNAL_VERSION = 1

_PARA_SPLIT = re.compile(r"\n\s*\n")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'\u2019-][a-z0-9]+)*", re.IGNORECASE)
_TRANSCRIPT_FORM = "TRANSCRIPT"


@dataclass
class DiffItem:
    kind: str
    item: str
    similarity: float
    old_text: str
    new_text: str


@dataclass(frozen=True)
class DocumentPair:
    """A document and the strictly earlier document it can be compared with."""

    current: Document
    previous: Document | None


@dataclass(frozen=True)
class DisclosureSignal:
    """Normalized lexical disclosure change for one document.

    ``score`` is in [0, 1]: zero means that the section-aligned token
    multisets are identical and one means they have no token overlap.  A
    missing comparison is represented by ``None`` rather than zero so a
    regression cannot accidentally treat an unavailable signal as "no
    change".
    """

    doc_id: str
    ticker: str
    form: str
    date: str
    predecessor_doc_id: str | None
    score: float | None
    changed_token_mass: int = 0
    comparison_token_mass: int = 0
    substantive_token_mass: int = 0
    new_removed_token_mass: int = 0
    minor_token_mass: int = 0
    sections_added: int = 0
    sections_removed: int = 0
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.score is not None

    @property
    def disclosure_change(self) -> float | None:
        """Regression-facing name for the normalized score."""
        return self.score


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
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(settings.embedding_model)
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
            results.append(DiffItem(kind, new_sec.item, round(score, 3),
                                    old_paras[i][:600] if i >= 0 else "", np_text[:600]))

        for i, op in enumerate(old_paras):
            if i not in matched_old and len(op) > 80:
                results.append(DiffItem("removed", new_sec.item, 0.0, op[:600], ""))

    order = {"new": 0, "substantive": 1, "removed": 2, "minor": 3}
    results.sort(key=lambda d: order[d.kind])
    return results[:max_items]


# --- corpus-level numeric disclosure-change signal ------------------------
#
# The interactive diff above may use embeddings because a person benefits
# from seeing semantically similar paragraphs next to each other.  A
# regression feature has different requirements: it must be reproducible,
# bounded, quick to calculate for the whole corpus, and must never vary with
# an optional ML dependency.  The implementation below therefore applies the
# same lexical fallback categories as ``diff_documents`` (new, substantive,
# minor, boilerplate) to section-aligned token multisets.

_LEXICAL_NEW_THRESHOLD = 0.30
_LEXICAL_SUBSTANTIVE_THRESHOLD = 0.65
_LEXICAL_BOILERPLATE_THRESHOLD = 0.90
_MINOR_CHANGE_WEIGHT = 0.25


@dataclass
class _DocumentProfile:
    sections: dict[str, Counter[str]]
    token_count: int


@dataclass(frozen=True)
class _ChangeSummary:
    score: float | None
    changed: int
    comparison: int
    substantive: int
    new_removed: int
    minor: int
    sections_added: int
    sections_removed: int


def _document_sort_key(document: Document) -> tuple[str, str, str, str, str]:
    return (
        document.ticker.upper(),
        document.form.upper(),
        document.date,
        document.doc_id,
        str(document.path),
    )


def _iter_document_groups(
    documents: Iterable[Document],
) -> Iterable[tuple[tuple[str, str], list[Document]]]:
    ordered = sorted(documents, key=_document_sort_key)
    for key, group in groupby(
        ordered, key=lambda d: (d.ticker.upper(), d.form.upper())
    ):
        yield key, list(group)


def pair_documents_with_previous(documents: Iterable[Document]) -> list[DocumentPair]:
    """Pair every document with the previous same-ticker/same-form filing.

    "Previous" is strict: two files dated on the same day never become one
    another's predecessor.  If duplicate files exist for a date, the
    lexicographically first path is used as the deterministic predecessor for
    the next date.  Transcript pairs are returned by this low-level helper,
    but :func:`compute_disclosure_signals` marks them unsupported because
    speaker-turn changes are not comparable filing disclosures.
    """
    pairs: list[DocumentPair] = []
    for _key, group in _iter_document_groups(documents):
        previous: Document | None = None
        for _date, same_date_iter in groupby(group, key=lambda d: d.date):
            same_date = list(same_date_iter)
            pairs.extend(DocumentPair(document, previous) for document in same_date)
            previous = same_date[0]
    return pairs


def _profile_document(document: Document) -> _DocumentProfile:
    """Build a case/punctuation-normalized token multiset for each Item."""
    sections: dict[str, Counter[str]] = {}
    token_count = 0
    for section in document.sections:
        # Duplicate Item headings are unusual but combining them makes this
        # robust to hand-built Documents as well as the ingestion parser.
        item = section.item.strip().upper() or "0"
        tokens = _TOKEN.findall(section.text.casefold())
        if not tokens:
            continue
        counter = sections.setdefault(item, Counter())
        counter.update(tokens)
        token_count += len(tokens)
    return _DocumentProfile(sections, token_count)


def _summarize_change(old: _DocumentProfile, new: _DocumentProfile) -> _ChangeSummary:
    """Turn aligned section changes into one normalized weighted score.

    For a section present in both filings we use multiset Sorensen-Dice
    similarity.  Its changed token mass is then assigned the same categories
    used by the lexical diff engine:

    * similarity < .30: replacement/new-removed change (weight 1.0)
    * .30 <= similarity < .65: substantive change (weight 1.0)
    * .65 <= similarity < .90: minor change (weight 0.25)
    * similarity >= .90: boilerplate-equivalent (weight 0.0)

    Entirely added or removed sections receive weight 1.0.  Dividing by the
    combined old/new token mass bounds the result to [0, 1] and prevents long
    filings from receiving mechanically larger values.
    """
    comparison = old.token_count + new.token_count
    if comparison == 0:
        return _ChangeSummary(None, 0, 0, 0, 0, 0, 0, 0)

    substantive = new_removed = minor = 0
    sections_added = sections_removed = 0

    for item in old.sections.keys() | new.sections.keys():
        old_tokens = old.sections.get(item)
        new_tokens = new.sections.get(item)
        if old_tokens is None:
            mass = new_tokens.total() if new_tokens is not None else 0
            new_removed += mass
            sections_added += 1
            continue
        if new_tokens is None:
            new_removed += old_tokens.total()
            sections_removed += 1
            continue

        old_n, new_n = old_tokens.total(), new_tokens.total()
        section_mass = old_n + new_n
        if section_mass == 0:
            continue
        overlap = sum((old_tokens & new_tokens).values())
        similarity = (2.0 * overlap) / section_mass
        changed_mass = section_mass - (2 * overlap)

        if similarity < _LEXICAL_NEW_THRESHOLD:
            new_removed += changed_mass
        elif similarity < _LEXICAL_SUBSTANTIVE_THRESHOLD:
            substantive += changed_mass
        elif similarity < _LEXICAL_BOILERPLATE_THRESHOLD:
            minor += changed_mass

    changed = substantive + new_removed + minor
    weighted = substantive + new_removed + (_MINOR_CHANGE_WEIGHT * minor)
    score = round(weighted / comparison, 12)
    # Floating-point arithmetic cannot make this exceed the bounds in normal
    # operation, but clipping documents the public invariant defensively.
    score = min(1.0, max(0.0, score))
    return _ChangeSummary(
        score,
        changed,
        comparison,
        substantive,
        new_removed,
        minor,
        sections_added,
        sections_removed,
    )


def _unavailable_signal(document: Document, reason: str) -> DisclosureSignal:
    return DisclosureSignal(
        doc_id=document.doc_id,
        ticker=document.ticker,
        form=document.form,
        date=document.date,
        predecessor_doc_id=None,
        score=None,
        reason=reason,
    )


def compute_disclosure_signals(documents: Iterable[Document]) -> list[DisclosureSignal]:
    """Compute deterministic disclosure-change signals for a whole corpus.

    Profiles are built once per document and only the previous date's profile
    is retained, making the calculation linear in corpus text size (apart
    from the initial sort) and suitable for the current 1,255-document corpus.
    Transcripts intentionally receive ``None`` because speaker/participant
    turnover is not comparable to Item-level SEC disclosure changes.
    """
    signals: list[DisclosureSignal] = []
    for (_ticker, form), group in _iter_document_groups(documents):
        if form == _TRANSCRIPT_FORM:
            signals.extend(
                _unavailable_signal(document, "unsupported_transcript")
                for document in group
            )
            continue

        previous_document: Document | None = None
        previous_profile: _DocumentProfile | None = None
        for _date, same_date_iter in groupby(group, key=lambda d: d.date):
            same_date = list(same_date_iter)
            current_profiles = [
                (document, _profile_document(document)) for document in same_date
            ]
            for document, profile in current_profiles:
                if previous_document is None or previous_profile is None:
                    signals.append(_unavailable_signal(document, "no_predecessor"))
                    continue
                summary = _summarize_change(previous_profile, profile)
                if summary.score is None:
                    signals.append(
                        DisclosureSignal(
                            doc_id=document.doc_id,
                            ticker=document.ticker,
                            form=document.form,
                            date=document.date,
                            predecessor_doc_id=previous_document.doc_id,
                            score=None,
                            reason="empty_comparison",
                        )
                    )
                    continue
                signals.append(
                    DisclosureSignal(
                        doc_id=document.doc_id,
                        ticker=document.ticker,
                        form=document.form,
                        date=document.date,
                        predecessor_doc_id=previous_document.doc_id,
                        score=summary.score,
                        changed_token_mass=summary.changed,
                        comparison_token_mass=summary.comparison,
                        substantive_token_mass=summary.substantive,
                        new_removed_token_mass=summary.new_removed,
                        minor_token_mass=summary.minor,
                        sections_added=summary.sections_added,
                        sections_removed=summary.sections_removed,
                    )
                )

            # Same-date duplicates all compare with the same strictly earlier
            # document; the first sorted path is the canonical next predecessor.
            previous_document, previous_profile = current_profiles[0]

    return signals


def disclosure_signal_map(documents: Iterable[Document]) -> dict[str, float | None]:
    """Return the regression-ready ``doc_id -> disclosure_change`` mapping."""
    return {
        signal.doc_id: signal.disclosure_change
        for signal in compute_disclosure_signals(documents)
    }
