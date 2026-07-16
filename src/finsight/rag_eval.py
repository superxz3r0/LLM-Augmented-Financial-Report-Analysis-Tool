"""Lightweight and reproducible RAG evaluation utilities."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    # Support direct execution:
    #   python src/finsight/rag_eval.py
    # Relative imports only work when Python knows this file belongs to the
    # finsight package, so direct execution needs src/ on sys.path first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from finsight.audit import audit_answer
    from finsight.config import FILINGS_DIR, INDEX_DIR, PROJECT_ROOT, SAMPLE_DIR, settings
    from finsight.rag import RagAnswer, answer as rag_answer, extract_query_filters
else:
    from .audit import audit_answer
    from .config import FILINGS_DIR, INDEX_DIR, PROJECT_ROOT, SAMPLE_DIR, settings
    from .rag import RagAnswer, answer as rag_answer, extract_query_filters

DEFAULT_EVAL_PATH = PROJECT_ROOT / "eval" / "rag_questions.json"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "eval" / "rag_eval_results.json"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "eval" / "rag_eval_results.csv"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "eval" / "rag_eval_summary.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "eval" / "rag_eval_report.md"
EVAL_TMP_DIR = PROJECT_ROOT / "eval" / "tmp"
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"
RELEVANT_CONTEXT_THRESHOLD = 0.70
DEFAULT_PASS_THRESHOLD = 0.77
MIN_CONTEXTUAL_RECALL = 0.60
MIN_CONTEXTUAL_PRECISION = 0.20
MIN_EVIDENCE_HIT = 0.60
MIN_MRR = 0.20

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_CITE = re.compile(r"\[(\d+)\]")
_SPACE = re.compile(r"\s+")
_PERCENT = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent)\b", re.IGNORECASE)
_AMOUNT = re.compile(
    r"(?<![a-z0-9])(?:\$|us\$|usd\s*)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(trillion|billion|million|bn|m)?\b",
    re.IGNORECASE,
)
_ABSTAIN = re.compile(
    r"\b(no relevant|not contain|does not contain|do not contain|not provide|"
    r"cannot determine|cannot confirm|can't determine|insufficient|unable to|"
    r"not available|not found|do not have|does not provide)\b",
    re.IGNORECASE,
)

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "into", "what",
    "does", "did", "how", "its", "our", "are", "was", "were", "has",
    "have", "had", "will", "would", "could", "about", "according", "give",
    "provided", "provide", "filing", "fiscal", "form",
    "usd", "us", "million", "billion", "trillion", "bn", "percent",
}

CSV_FIELDS = [
    "question_id",
    "question_type",
    "ticker",
    "question",
    "expected_unanswerable",
    "passed",
    "overall_score",
    "answer_correctness",
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "citation_correctness",
    "evidence_hit",
    "exact_chunk_hit",
    "mrr",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "failure_category",
    "failure_reason",
    "failed_metrics",
    "retrieved_chunk_ids",
    "gold_chunk_ids",
    "answer_backend",
    "evaluation_mode",
    "index_reused",
    "embeddings_regenerated",
]


@dataclass(frozen=True)
class GoldSource:
    ticker: str = ""
    form: str = ""
    date: str = ""
    item: str = ""
    section: str = ""
    evidence: str = ""
    source_file: str = ""
    doc_id: str = ""
    chunk_id: str = ""
    acceptable_chunk_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldSource":
        return cls(
            ticker=data.get("ticker", ""),
            form=data.get("form", ""),
            date=data.get("date", ""),
            item=str(data.get("item", "")),
            section=data.get("section", ""),
            evidence=data.get("evidence", ""),
            source_file=data.get("source_file", ""),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id", ""),
            acceptable_chunk_ids=tuple(
                str(item) for item in data.get("acceptable_chunk_ids", [])
            ),
        )


@dataclass(frozen=True)
class RagEvalCase:
    question_id: str
    question: str
    gold_answer: str
    gold_sources: list[GoldSource]
    question_type: str = ""
    difficulty: str = ""
    ticker: str = ""
    company: str = ""
    filing_type: str = ""
    filing_date: str = ""
    target_item: str = ""
    target_section: str = ""
    metric_focus: list[str] = field(default_factory=list)
    expected_unanswerable: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagEvalCase":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            gold_answer=data.get("gold_answer", ""),
            gold_sources=[GoldSource.from_dict(s) for s in data.get("gold_sources", [])],
            question_type=data.get("question_type", ""),
            difficulty=data.get("difficulty", ""),
            ticker=data.get("ticker", ""),
            company=data.get("company", ""),
            filing_type=data.get("filing_type", ""),
            filing_date=data.get("filing_date", ""),
            target_item=str(data.get("target_item", "")),
            target_section=data.get("target_section", ""),
            metric_focus=list(data.get("metric_focus", [])),
            expected_unanswerable=bool(data.get("expected_unanswerable", False)),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class MetricScores:
    answer_relevancy: float | None
    faithfulness: float | None
    contextual_relevancy: float | None
    contextual_recall: float | None
    contextual_precision: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "answer_relevancy": self.answer_relevancy,
            "faithfulness": self.faithfulness,
            "contextual_relevancy": self.contextual_relevancy,
            "contextual_recall": self.contextual_recall,
            "contextual_precision": self.contextual_precision,
        }


@dataclass(frozen=True)
class RagEvalResult:
    case: RagEvalCase
    answer: RagAnswer
    metrics: MetricScores
    diagnostics: dict[str, float | None] = field(default_factory=dict)

    @property
    def overall_score(self) -> float | None:
        return _overall_score(self)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question_id": self.case.question_id,
            "type": self.case.question_type,
            "difficulty": self.case.difficulty,
            "expected_unanswerable": self.case.expected_unanswerable,
            "overall": self.overall_score,
            "failure_category": self.answer.failure_category,
            "error_message": self.answer.error_message,
        }
        row.update(self.metrics.as_dict())
        row.update(self.diagnostics)
        return row


Answerer = Callable[..., RagAnswer]


def load_eval_set(path: Path | None = None) -> list[RagEvalCase]:
    data = json.loads((path or DEFAULT_EVAL_PATH).read_text(encoding="utf-8-sig"))
    return [RagEvalCase.from_dict(item) for item in data]


def evaluate_case(index: Any, case: RagEvalCase, k: int | None = None,
                  answerer: Answerer | None = None) -> RagEvalResult:
    k = k or settings.top_k
    filters = _case_filters(case)
    if answerer is None:
        ans = _default_answerer(index, case.question, k, ticker=case.ticker,
                                filters=filters)
    else:
        try:
            ans = answerer(index, case.question, k, case.ticker, filters)
        except TypeError:
            try:
                ans = answerer(index, case.question, k, case.ticker)
            except TypeError:
                ans = answerer(index, case.question, k)
    if ans.retrieval_sources is not None:
        retrieved = ans.retrieval_sources
    elif ans.sources:
        retrieved = ans.sources
    else:
        retrieved = _search_index(index, case.question, k, case.ticker, filters)

    answer_relevancy = _answer_relevancy(ans.text, case)
    faithfulness = _faithfulness(ans.text, retrieved, case)
    contextual_relevancy = _contextual_relevancy(retrieved, case)
    contextual_recall = _contextual_recall(retrieved, case)
    contextual_precision = _contextual_precision(retrieved, case)

    metrics = MetricScores(
        answer_relevancy=answer_relevancy,
        faithfulness=faithfulness,
        contextual_relevancy=contextual_relevancy,
        contextual_recall=contextual_recall,
        contextual_precision=contextual_precision,
    )
    diagnostics = {
        "recall_at_k": contextual_recall,
        "recall_at_1": _recall_at_n(retrieved, case, 1),
        "recall_at_3": _recall_at_n(retrieved, case, 3),
        "recall_at_5": _recall_at_n(retrieved, case, 5),
        "mrr": _mrr(retrieved, case),
        "exact_chunk_hit": _exact_chunk_hit(retrieved, case),
        "evidence_hit": _evidence_hit(retrieved, case),
        "retrieval_hit": contextual_recall,
        "metadata_hit_rate": _metadata_hit_rate(retrieved, case),
        "citation_presence": _citation_presence(ans.text, case),
        "citation_correctness": _citation_correctness(ans.text, retrieved, case),
        "answer_correctness": _answer_correctness(ans.text, case),
        "hallucination_rate": None if faithfulness is None else round(1.0 - faithfulness, 3),
        "abstention_accuracy": _abstention_accuracy(ans.text, case),
    }
    return RagEvalResult(case=case, answer=ans, metrics=metrics,
                         diagnostics={k: _round(v) for k, v in diagnostics.items()})


def run_evaluation(index: Any, cases: list[RagEvalCase] | None = None,
                   k: int | None = None,
                   answerer: Answerer | None = None) -> list[RagEvalResult]:
    return [evaluate_case(index, case, k=k, answerer=answerer)
            for case in (cases or load_eval_set())]


def summarize_results(results: list[RagEvalResult]) -> dict[str, Any]:
    metric_keys = MetricScores(None, None, None, None, None).as_dict().keys()
    diagnostic_keys = sorted({k for r in results for k in r.diagnostics})
    metrics = {
        key: _round(_mean(r.metrics.as_dict()[key] for r in results
                          if r.metrics.as_dict()[key] is not None))
        for key in metric_keys
    }
    diagnostics = {
        key: _round(_mean(r.diagnostics.get(key) for r in results
                          if r.diagnostics.get(key) is not None))
        for key in diagnostic_keys
    }
    answerable = [r for r in results if not r.case.expected_unanswerable]
    unanswerable = [r for r in results if r.case.expected_unanswerable]
    return {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "overall": _round(_mean(r.overall_score for r in results)),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _default_answerer(index: Any, question: str, k: int,
                      ticker: str | None = None,
                      filters: dict[str, Any] | None = None) -> RagAnswer:
    return rag_answer(index, question, ticker=ticker, k=k, filters=filters)


def _search_index(index: Any, question: str, k: int,
                  ticker: str | None = None,
                  filters: dict[str, Any] | None = None) -> list:
    filters = _merge_query_filters(question, filters)
    if ticker:
        try:
            return index.search(question, k, ticker=ticker, **filters)
        except TypeError:
            pass
    try:
        return index.search(question, k, **filters)
    except TypeError:
        return index.search(question, k)


def _case_filters(case: RagEvalCase) -> dict[str, list[str] | str]:
    # Use only metadata stated in the question. Gold-source metadata is used
    # for scoring, not for helping retrieval find the answer.
    return extract_query_filters(case.question)


def _merge_query_filters(question: str, filters: dict[str, Any] | None) -> dict[str, Any]:
    merged = extract_query_filters(question)
    if filters:
        for key, value in filters.items():
            if value not in (None, "", [], ()):
                merged[key] = value
    return merged


def _normalise(text: str) -> str:
    return _SPACE.sub(" ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    tokens = {t for t in _WORD.findall(text.lower()) if t not in _STOP}
    tokens.update(_numeric_tokens(text))
    return tokens


def _numeric_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for match in _PERCENT.finditer(text):
        value = _parse_decimal(match.group(1))
        if value is not None:
            out.add(f"pct:{value:g}")
    for match in _AMOUNT.finditer(text):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        value = _parse_decimal(raw)
        if value is None:
            continue
        if unit in {"trillion"}:
            out.add(f"usd_m:{value * 1_000_000:g}")
        elif unit in {"billion", "bn"}:
            out.add(f"usd_m:{value * 1_000:g}")
        elif unit in {"million", "m"}:
            out.add(f"usd_m:{value:g}")
        else:
            out.add(f"num:{value:g}")
    return out


def _parse_decimal(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _coverage_score(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    if _normalise(reference) in _normalise(candidate):
        return 1.0
    cand, ref = _tokens(candidate), _tokens(reference)
    if not cand or not ref:
        return 0.0
    overlap = len(cand & ref)
    recall = overlap / len(ref)
    precision = overlap / len(cand)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return max(f1, 0.85 * recall)


def _answer_relevancy(answer_text: str, case: RagEvalCase) -> float:
    if case.expected_unanswerable:
        return 1.0 if _is_abstention(answer_text) else 0.0
    return _round(_normalised_question_alignment(answer_text, case)) or 0.0


def _answer_correctness(answer_text: str, case: RagEvalCase) -> float:
    if case.expected_unanswerable:
        return 1.0 if _is_abstention(answer_text) else 0.0
    if len(case.gold_sources) > 1:
        # A comparison answer must cover every filing rather than matching one
        # evidence span perfectly and ignoring the other filings.
        return _round(_mean(
            _coverage_score(answer_text, source.evidence)
            for source in case.gold_sources
            if source.evidence
        )) or 0.0
    references = [case.gold_answer] + [s.evidence for s in case.gold_sources]
    return _round(max(
        (_coverage_score(answer_text, reference) for reference in references if reference),
        default=0.0,
    )) or 0.0


def _faithfulness(answer_text: str, retrieved: list, case: RagEvalCase) -> float:
    if case.expected_unanswerable:
        return 1.0 if _is_abstention(answer_text) else 0.0
    chunks = [chunk for chunk, _ in retrieved]
    report = audit_answer(answer_text, chunks)
    return _round(report.grounded_ratio)


def _contextual_relevancy(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    scores = [_chunk_question_relevance(chunk, case) for chunk, _ in retrieved]
    return _round(_mean(scores))


def _contextual_recall(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    covered = 0
    for source in case.gold_sources:
        if any(_source_match_score(chunk, source) >= RELEVANT_CONTEXT_THRESHOLD
               for chunk, _ in retrieved):
            covered += 1
    return _round(covered / len(case.gold_sources))


def _recall_at_n(retrieved: list, case: RagEvalCase, n: int) -> float | None:
    return _contextual_recall(retrieved[:n], case)


def _contextual_precision(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    if not retrieved:
        return 0.0
    relevant = sum(
        1
        for chunk, _ in retrieved
        if _chunk_gold_relevance(chunk, case) >= RELEVANT_CONTEXT_THRESHOLD
    )
    return _round(relevant / len(retrieved))


def _mrr(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    for rank, (chunk, _) in enumerate(retrieved, 1):
        if _chunk_gold_relevance(chunk, case) >= RELEVANT_CONTEXT_THRESHOLD:
            return _round(1 / rank)
    return 0.0


def _exact_chunk_hit(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    hits = 0
    for source in case.gold_sources:
        if any(_chunk_id_score(chunk, source) >= 1.0 for chunk, _ in retrieved):
            hits += 1
    return _round(hits / len(case.gold_sources))


def _evidence_hit(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    hits = 0
    for source in case.gold_sources:
        if any(_coverage_score(getattr(chunk, "text", ""), source.evidence) >= 0.70
               for chunk, _ in retrieved):
            hits += 1
    return _round(hits / len(case.gold_sources))


def _metadata_hit_rate(retrieved: list, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable or not case.gold_sources:
        return None
    covered = 0
    for source in case.gold_sources:
        if any(_metadata_score(chunk, source) >= 0.8 for chunk, _ in retrieved):
            covered += 1
    return _round(covered / len(case.gold_sources))


def _citation_presence(answer_text: str, case: RagEvalCase) -> float | None:
    if case.expected_unanswerable:
        return None
    return 1.0 if _CITE.search(answer_text) else 0.0


def _citation_correctness(answer_text: str, retrieved: list, case: RagEvalCase) -> float | None:
    cited = [int(n) for n in _CITE.findall(answer_text)]
    if case.expected_unanswerable:
        return 1.0 if _is_abstention(answer_text) and not cited else 0.0
    if not cited:
        return 0.0
    correct = 0
    for n in cited:
        if not 1 <= n <= len(retrieved):
            continue
        chunk, _ = retrieved[n - 1]
        if _chunk_gold_relevance(chunk, case) >= RELEVANT_CONTEXT_THRESHOLD:
            correct += 1
    return _round(correct / len(cited))


def _abstention_accuracy(answer_text: str, case: RagEvalCase) -> float | None:
    if not case.expected_unanswerable:
        return None
    return 1.0 if _is_abstention(answer_text) else 0.0


def _is_abstention(answer_text: str) -> bool:
    return bool(_ABSTAIN.search(answer_text))


def _chunk_question_relevance(chunk: Any, case: RagEvalCase) -> float:
    return _normalised_question_alignment(chunk.text, case)


def _normalised_question_alignment(candidate: str, case: RagEvalCase) -> float:
    raw_score = _question_alignment(candidate, case)
    references = [case.gold_answer] + [source.evidence for source in case.gold_sources]
    reference_score = max(
        (_question_alignment(reference, case) for reference in references if reference),
        default=0.0,
    )
    # Gold evidence is used only to estimate how much of the question wording
    # a relevant passage can reasonably contain.
    if reference_score > 0:
        return min(raw_score / reference_score, 1.0)
    return raw_score


def _question_alignment(candidate: str, case: RagEvalCase) -> float:
    question_terms = _tokens(case.question)
    metadata = " ".join(filter(None, [
        case.ticker,
        case.company,
        case.filing_type,
        case.filing_date,
        case.target_item,
        case.target_section,
    ]))
    focused_terms = question_terms - _tokens(metadata)
    if not focused_terms:
        focused_terms = question_terms
    if not focused_terms:
        return 0.0
    return len(_tokens(candidate) & focused_terms) / len(focused_terms)


def _chunk_gold_relevance(chunk: Any, case: RagEvalCase) -> float:
    return max((_source_match_score(chunk, s) for s in case.gold_sources), default=0.0)


def _source_match_score(chunk: Any, source: GoldSource) -> float:
    id_score = _chunk_id_score(chunk, source)
    if id_score >= 1.0:
        return 1.0
    if id_score >= 0.90:
        return id_score
    evidence = _coverage_score(chunk.text, source.evidence)
    metadata = _metadata_score(chunk, source)
    return max(id_score, 0.80 * evidence + 0.20 * metadata)


def _chunk_id_score(chunk: Any, source: GoldSource) -> float:
    chunk_id = str(getattr(chunk, "chunk_id", ""))
    if not chunk_id:
        return 0.0
    if source.chunk_id and chunk_id == source.chunk_id:
        return 1.0
    acceptable = set(source.acceptable_chunk_ids)
    if acceptable and chunk_id in acceptable:
        return 0.92
    if source.doc_id and str(getattr(chunk, "doc_id", "")) == source.doc_id:
        source_num = _chunk_number(source.chunk_id)
        chunk_num = _chunk_number(chunk_id)
        if source_num is not None and chunk_num is not None:
            distance = abs(source_num - chunk_num)
            if distance == 1:
                return 0.88
            if distance == 2:
                return 0.75
    return 0.0


def _chunk_number(chunk_id: str) -> int | None:
    match = re.search(r"#(\d+)$", chunk_id or "")
    if not match:
        return None
    return int(match.group(1))


def _metadata_score(chunk: Any, source: GoldSource) -> float:
    checks: list[bool] = []
    for field in ("ticker", "form", "date", "item"):
        expected = getattr(source, field)
        if expected:
            checks.append(str(getattr(chunk, field, "")).lower() == str(expected).lower())
    if source.section:
        actual = _normalise(getattr(chunk, "section_title", ""))
        expected_section = _normalise(source.section)
        checks.append(expected_section in actual or actual in expected_section)
    if not checks:
        return 0.0
    return sum(checks) / len(checks)


def result_rows(results: list[RagEvalResult],
                pass_threshold: float = DEFAULT_PASS_THRESHOLD,
                index_info: dict[str, Any] | None = None,
                evaluation_mode: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index_info = index_info or {}
    for result in results:
        passed = case_passed(result, pass_threshold)
        row = {
            "question_id": result.case.question_id,
            "question_type": result.case.question_type,
            "ticker": result.case.ticker,
            "question": result.case.question,
            "expected_unanswerable": result.case.expected_unanswerable,
            "passed": passed,
            "overall_score": result.overall_score,
            "answer_correctness": result.diagnostics.get("answer_correctness"),
            "answer_relevancy": result.metrics.answer_relevancy,
            "faithfulness": result.metrics.faithfulness,
            "contextual_precision": result.metrics.contextual_precision,
            "contextual_recall": result.metrics.contextual_recall,
            "citation_correctness": result.diagnostics.get("citation_correctness"),
            "evidence_hit": result.diagnostics.get("evidence_hit"),
            "exact_chunk_hit": result.diagnostics.get("exact_chunk_hit"),
            "mrr": result.diagnostics.get("mrr"),
            "recall_at_1": result.diagnostics.get("recall_at_1"),
            "recall_at_3": result.diagnostics.get("recall_at_3"),
            "recall_at_5": result.diagnostics.get("recall_at_5"),
            "failure_category": _failure_category(result, passed),
            "failure_reason": _failure_reason(result, passed, pass_threshold),
            "failed_metrics": _failed_metrics(result, pass_threshold),
            "retrieved_chunk_ids": _retrieved_chunk_ids(result),
            "gold_chunk_ids": _gold_chunk_ids(result.case),
            "answer_backend": result.answer.backend,
            "evaluation_mode": evaluation_mode,
            "index_reused": bool(index_info.get("index_reused", False)),
            "embeddings_regenerated": bool(index_info.get("embeddings_regenerated", False)),
        }
        rows.append(row)
    return rows


def case_passed(result: RagEvalResult,
                pass_threshold: float = DEFAULT_PASS_THRESHOLD) -> bool:
    if result.case.expected_unanswerable:
        return result.diagnostics.get("abstention_accuracy") == 1.0
    required = [
        (result.metrics.contextual_recall, MIN_CONTEXTUAL_RECALL),
        (result.metrics.contextual_precision, MIN_CONTEXTUAL_PRECISION),
        (result.diagnostics.get("evidence_hit"), MIN_EVIDENCE_HIT),
        (result.diagnostics.get("mrr"), MIN_MRR),
    ]
    return (
        result.overall_score is not None
        and result.overall_score >= pass_threshold
        and all(value is not None and value >= threshold
                for value, threshold in required)
    )


def pass_fail_summary(results: list[RagEvalResult],
                      pass_threshold: float = DEFAULT_PASS_THRESHOLD,
                      index_info: dict[str, Any] | None = None,
                      evaluation_mode: str = "") -> dict[str, Any]:
    rows = result_rows(results, pass_threshold, index_info, evaluation_mode)
    type_counts = Counter(row["question_type"] for row in rows)
    type_pass = Counter(row["question_type"] for row in rows if row["passed"])
    passed = sum(1 for row in rows if row["passed"])
    failed = len(rows) - passed
    answerable = [row for row in rows if not row["expected_unanswerable"]]
    unanswerable = [row for row in rows if row["expected_unanswerable"]]
    return {
        "pass_threshold": pass_threshold,
        "passed": passed,
        "failed": failed,
        "total": len(rows),
        "pass_rate": _round(_ratio(passed, len(rows))),
        "answerable_pass_rate": _round(_ratio(
            sum(1 for row in answerable if row["passed"]), len(answerable)
        )),
        "unanswerable_pass_rate": _round(_ratio(
            sum(1 for row in unanswerable if row["passed"]), len(unanswerable)
        )),
        "answer_backend_counts": dict(Counter(row["answer_backend"] for row in rows)),
        "fallback_after_provider_failure": sum(
            1
            for result in results
            if result.answer.backend == "extractive" and result.answer.failure_category
        ),
        "answer_failure_category_counts": dict(Counter(
            result.answer.failure_category
            for result in results
            if result.answer.failure_category
        )),
        "failure_category_counts": dict(
            Counter(row["failure_category"] for row in rows if not row["passed"])
        ),
        "summary": summarize_results(results),
        "by_type": {
            name: {
                "total": type_counts[name],
                "passed": type_pass[name],
                "failed": type_counts[name] - type_pass[name],
                "pass_rate": _round(_ratio(type_pass[name], type_counts[name])),
            }
            for name in sorted(type_counts)
        },
    }


def write_result_files(results: list[RagEvalResult],
                       pass_threshold: float = DEFAULT_PASS_THRESHOLD,
                       results_json: Path = DEFAULT_RESULTS_JSON,
                       results_csv: Path = DEFAULT_RESULTS_CSV,
                       summary_json: Path = DEFAULT_SUMMARY_JSON,
                       report_md: Path = DEFAULT_REPORT_MD,
                       extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    index_info = (extra_summary or {}).get("index", {})
    evaluation_mode = (extra_summary or {}).get("evaluation_mode", "")
    rows = result_rows(results, pass_threshold, index_info, evaluation_mode)
    summary = pass_fail_summary(results, pass_threshold, index_info, evaluation_mode)
    if extra_summary:
        summary.update(extra_summary)

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    if rows:
        with results_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        results_csv.write_text("", encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    write_markdown_report(results, rows, summary, report_md)
    return summary


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _retrieved_chunk_ids(result: RagEvalResult) -> str:
    return ";".join(
        str(getattr(chunk, "chunk_id", ""))
        for chunk, _ in result.answer.sources
        if getattr(chunk, "chunk_id", "")
    )


def _gold_chunk_ids(case: RagEvalCase) -> str:
    ids: list[str] = []
    for source in case.gold_sources:
        if source.chunk_id:
            ids.append(source.chunk_id)
        ids.extend(source.acceptable_chunk_ids)
    return ";".join(dict.fromkeys(ids))


def _failed_metrics(result: RagEvalResult,
                    pass_threshold: float = DEFAULT_PASS_THRESHOLD) -> str:
    if case_passed(result, pass_threshold):
        return ""
    if result.case.expected_unanswerable:
        checks = [
            ("abstention_accuracy", result.diagnostics.get("abstention_accuracy"), 1.0),
        ]
    else:
        checks = [
            ("overall_score", result.overall_score, pass_threshold),
            ("contextual_recall", result.metrics.contextual_recall,
             MIN_CONTEXTUAL_RECALL),
            ("contextual_precision", result.metrics.contextual_precision,
             MIN_CONTEXTUAL_PRECISION),
            ("evidence_hit", result.diagnostics.get("evidence_hit"), MIN_EVIDENCE_HIT),
            ("mrr", result.diagnostics.get("mrr"), MIN_MRR),
        ]
    failed = [
        (f"{name}=n/a (<{threshold:.2f})" if value is None
         else f"{name}={value:.3f} (<{threshold:.2f})")
        for name, value, threshold in checks
        if value is None or value < threshold
    ]
    return "; ".join(failed)


def _failure_category(result: RagEvalResult, passed: bool) -> str:
    if passed:
        return ""
    if result.answer.failure_category == "configuration_failure" and result.answer.backend != "extractive":
        return "Configuration failure"
    if result.answer.failure_category and result.answer.backend != "extractive":
        return "Generation failure"
    if result.diagnostics.get("recall_at_k") == 0.0:
        return "Retrieval failure"
    if result.overall_score is None:
        return "Evaluation failure"
    return "Overall quality failure"


def _failure_reason(result: RagEvalResult, passed: bool,
                    pass_threshold: float = DEFAULT_PASS_THRESHOLD) -> str:
    if passed:
        return ""
    failed = _failed_metrics(result, pass_threshold)
    if failed:
        return failed
    if result.answer.error_message:
        return str(result.answer.error_message).replace("\n", " ")[:240]
    if result.case.expected_unanswerable:
        return "Answer should abstain but did not."
    return "Overall score below threshold."


def _suggested_next_action(category: str) -> str:
    return {
        "Configuration failure": "Check API key/provider configuration.",
        "Index loading failure": "Inspect Chroma collection and manifest, then rebuild explicitly if needed.",
        "Retrieval failure": "Review chunking, metadata filters, and gold evidence coverage.",
        "Generation failure": "Inspect provider errors and retry policy.",
        "Evaluation failure": "Inspect evaluator inputs and metric null handling.",
        "Overall quality failure": "Review the retrieved chunks and the generated answer.",
    }.get(category, "Review this case manually.")


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _count_summary(counts: dict[str, int] | None) -> str:
    if not counts:
        return "None"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def _progress_bar(rate: float | None, width: int = 24) -> str:
    if rate is None:
        return "[" + "-" * width + "] n/a"
    filled = max(0, min(width, round(rate * width)))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {rate * 100:.1f}%"


def _md(text: Any, limit: int | None = None) -> str:
    value = "" if text is None else str(text)
    value = _SPACE.sub(" ", value).strip().replace("|", "\\|")
    if limit and len(value) > limit:
        value = value[: max(limit - 3, 0)].rstrip() + "..."
    return value


def write_markdown_report(
    results: list[RagEvalResult],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    report_md: Path,
) -> None:
    index_info = summary.get("index", {})
    metrics = summary.get("summary", {}).get("metrics", {})
    diagnostics = summary.get("summary", {}).get("diagnostics", {})
    pass_rate = summary.get("pass_rate")
    failed_rows = [row for row in rows if not row["passed"]]
    fragile_rows = sorted(
        (row for row in rows if row["passed"] and row.get("overall_score") is not None),
        key=lambda row: float(row["overall_score"]),
    )[:10]

    lines = [
        "# RAG Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Total cases | {summary.get('total', 0)} |",
        f"| Passed | {summary.get('passed', 0)} |",
        f"| Failed | {summary.get('failed', 0)} |",
        f"| Pass rate | {_pct(pass_rate)} |",
        f"| Primary score | Retrieval 85% + answer 15% |",
        f"| Answerable pass rate | {_pct(summary.get('answerable_pass_rate'))} |",
        f"| Unanswerable pass rate | {_pct(summary.get('unanswerable_pass_rate'))} |",
        f"| Evaluation mode | {_md(summary.get('evaluation_mode', ''))} |",
        f"| Actual answer backend(s) | {_md(_count_summary(summary.get('answer_backend_counts')))} |",
        f"| Fallbacks after provider failure | {summary.get('fallback_after_provider_failure', 0)} |",
        f"| Retrieval backend | {_md(summary.get('index_backend', ''))} |",
        f"| Existing index reused | {_yes_no(index_info.get('index_reused'))} |",
        f"| Embeddings regenerated | {_yes_no(index_info.get('embeddings_regenerated'))} |",
        f"| Evaluation timestamp | {_md(summary.get('evaluation_timestamp', ''))} |",
        "",
        f"Pass rate: {_progress_bar(pass_rate)}",
        "",
        "## Retrieval Metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Recall@1 | {_pct(diagnostics.get('recall_at_1'))} |",
        f"| Recall@3 | {_pct(diagnostics.get('recall_at_3'))} |",
        f"| Recall@5 | {_pct(diagnostics.get('recall_at_5'))} |",
        f"| MRR | {_score(diagnostics.get('mrr'))} |",
        f"| Evidence hit rate | {_pct(diagnostics.get('evidence_hit'))} |",
        f"| Exact chunk hit rate | {_pct(diagnostics.get('exact_chunk_hit'))} |",
        f"| Contextual precision | {_pct(metrics.get('contextual_precision'))} |",
        f"| Contextual recall | {_pct(metrics.get('contextual_recall'))} |",
        "",
        "## Answer Metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Answer correctness | {_pct(diagnostics.get('answer_correctness'))} |",
        f"| Answer relevancy | {_pct(metrics.get('answer_relevancy'))} |",
        f"| Faithfulness | {_pct(metrics.get('faithfulness'))} |",
        f"| Citation correctness | {_pct(diagnostics.get('citation_correctness'))} |",
        f"| Hallucination rate | {_pct(diagnostics.get('hallucination_rate'))} |",
        f"| Abstention accuracy | {_pct(diagnostics.get('abstention_accuracy'))} |",
        "",
        "## Results By Question Type",
        "",
        "| Question type | Total | Passed | Failed | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for question_type, data in summary.get("by_type", {}).items():
        lines.append(
            f"| {_md(question_type or 'Unspecified')} | {data['total']} | "
            f"{data['passed']} | {data['failed']} | {_pct(data.get('pass_rate'))} |"
        )

    lines.extend([
        "",
        "## Failure Category Summary",
        "",
        "| Category | Cases |",
        "|---|---:|",
    ])
    failure_counts = summary.get("failure_category_counts", {})
    if failure_counts:
        for category, count in sorted(failure_counts.items()):
            lines.append(f"| {_md(category)} | {count} |")
    else:
        lines.append("| None | 0 |")

    lines.extend([
        "",
        "## Failed Cases",
        "",
        "| ID | Type | Overall score | Failed metrics | Retrieved chunk(s) | Gold chunk(s) | Reason | Next action |",
        "|---|---|---:|---|---|---|---|---|",
    ])
    if failed_rows:
        for row in failed_rows:
            lines.append(
                f"| {_md(row['question_id'])} | {_md(row['question_type'])} | "
                f"{_score(row.get('overall_score'))} | {_md(row['failed_metrics'], 80)} | "
                f"{_md(row['retrieved_chunk_ids'], 80)} | {_md(row['gold_chunk_ids'], 80)} | "
                f"{_md(row['failure_reason'], 120)} | "
                f"{_md(_suggested_next_action(row['failure_category']), 80)} |"
            )
            lines.append(f"| | Question | | {_md(row['question'], 180)} | | | | |")
    else:
        lines.append("| None | | | | | | | |")

    lines.extend([
        "",
        "## Lowest-Scoring Passed Cases",
        "",
        "| ID | Type | Overall score | Failed/weak metrics |",
        "|---|---|---:|---|",
    ])
    if fragile_rows:
        for row in fragile_rows:
            lines.append(
                f"| {_md(row['question_id'])} | {_md(row['question_type'])} | "
                f"{_score(row.get('overall_score'))} | {_md(row.get('failed_metrics', ''), 120)} |"
            )
    else:
        lines.append("| None | | | |")

    lines.extend([
        "",
        "## Index Reuse Information",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Index path | {_md(index_info.get('index_path', ''))} |",
        f"| Index backend | {_md(index_info.get('backend', summary.get('index_backend', '')))} |",
        f"| Index status | {_md(index_info.get('index_status', ''))} |",
        f"| Manifest version | {_md(index_info.get('manifest_version', ''))} |",
        f"| Embedding model | {_md(index_info.get('embedding_model', ''))} |",
        f"| Embedding dimension | {_md(index_info.get('embedding_dimension', ''))} |",
        f"| Chunk configuration | {index_info.get('chunk_size', '')} / {index_info.get('chunk_overlap', '')} |",
        f"| Existing index reused | {_yes_no(index_info.get('index_reused'))} |",
        f"| Rebuild requested | {_yes_no(index_info.get('rebuild_requested'))} |",
        f"| Rebuild performed | {_yes_no(index_info.get('rebuild_performed'))} |",
        f"| Rebuild reason | {_md(index_info.get('rebuild_reason', ''))} |",
        f"| Index load time | {_score(index_info.get('load_time_seconds'))} s |",
        f"| Evaluation runtime | {_score(summary.get('evaluation_runtime_seconds'))} s |",
        f"| Restored read-only mtime touches | {_md(', '.join(summary.get('index_mtime_restored', [])))} |",
        f"| Unrestored index timestamp changes | {_md(', '.join(summary.get('index_mtime_changed', [])))} |",
        "",
        "## Reproduction Commands",
        "",
        "PowerShell:",
        "",
        "```powershell",
        "python -m pip install -r requirements.txt",
        "python -c \"import sys; sys.path.insert(0, 'src'); from finsight.llm import api_key_status; print(api_key_status('openai', 'OPENAI_API_KEY').message)\"",
        "pytest -q",
        "python src\\finsight\\rag_eval.py",
        "python src\\finsight\\rag_eval.py --min-pass-rate 0.85",
        "python src\\finsight\\rag_eval.py --with-llm",
        "python src\\finsight\\rag_eval.py --rebuild-index",
        "code eval\\rag_eval_report.md",
        "```",
        "",
        "Bash:",
        "",
        "```bash",
        "python -m pip install -r requirements.txt",
        "PYTHONPATH=src python -c \"from finsight.llm import api_key_status; print(api_key_status('openai', 'OPENAI_API_KEY').message)\"",
        "pytest -q",
        "python src/finsight/rag_eval.py",
        "python src/finsight/rag_eval.py --min-pass-rate 0.85",
        "python src/finsight/rag_eval.py --with-llm",
        "python src/finsight/rag_eval.py --rebuild-index",
        "${EDITOR:-vi} eval/rag_eval_report.md",
        "```",
        "",
    ])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def build_cli_index(bm25_only: bool = False,
                    rebuild_index: bool = False) -> tuple[Any, str, int, int, dict[str, Any]]:
    from finsight.chunker import chunk_corpus
    from finsight.index import _BM25, build_index
    from finsight.ingest import load_corpus

    # Match app.py so persisted Chroma fingerprints are identical. The eval
    # questions target data/filings, but keeping sample docs in the index avoids
    # invalidating an app-built index that includes the offline demo corpus.
    docs = load_corpus(FILINGS_DIR, SAMPLE_DIR)
    chunks = chunk_corpus(docs)
    if bm25_only:
        info = {
            "backend": "bm25",
            "index_path": "",
            "index_status": "bm25-only",
            "index_reused": False,
            "embeddings_regenerated": False,
            "rebuild_requested": rebuild_index,
            "rebuild_performed": False,
            "rebuild_reason": "",
            "manifest_version": None,
            "embedding_provider": "none",
            "embedding_model": "none",
            "embedding_dimension": None,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "doc_count": len(docs),
            "chunk_count": len(chunks),
            "corpus_fingerprint": "",
            "load_time_seconds": 0.0,
            "query_encoder_loaded": False,
            "query_encoder_error": "",
        }
        return _BM25(chunks), "bm25-only", len(docs), len(chunks), info

    index, backend = build_index(
        chunks,
        rebuild_index=rebuild_index,
        require_persistent=True,
        allow_tfidf_fallback=False,
        doc_count=len(docs),
    )
    return index, backend, len(docs), len(chunks), getattr(index, "index_info", {})


def extractive_answerer(index: Any, question: str, k: int,
                        ticker: str | None = None,
                        filters: dict[str, Any] | None = None) -> RagAnswer:
    from finsight.rag import _extractive_answer

    hits = _search_index(index, question, k, ticker, filters)
    if not hits:
        return RagAnswer("No relevant passages found in the corpus.", [],
                         "extractive")
    return _extractive_answer(question, hits)


def retrieval_only_answerer(index: Any, question: str, k: int,
                            ticker: str | None = None,
                            filters: dict[str, Any] | None = None) -> RagAnswer:
    from finsight.rag import _extractive_answer, retrieve_contexts

    hits, search_filters = retrieve_contexts(
        index, question, ticker=ticker, k=k, filters=filters
    )
    if not hits:
        return RagAnswer("No relevant passages found in the corpus.", [], "retrieval_only")
    answer = _extractive_answer(question, hits, search_filters)
    answer.backend = "retrieval_only"
    return answer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation set and write per-case scores."
    )
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH,
                        help="Path to rag_questions.json.")
    parser.add_argument("--top-k", type=int, default=settings.top_k,
                        help="Number of retrieved chunks to evaluate.")
    parser.add_argument("--pass-threshold", type=float,
                        default=DEFAULT_PASS_THRESHOLD,
                        help="Overall score required for a case to pass.")
    parser.add_argument("--min-pass-rate", type=float,
                        help="Return a non-zero exit code when the final pass rate is lower.")
    parser.add_argument("--limit", type=int,
                        help="Evaluate only the first N cases.")
    parser.add_argument("--question-id", action="append", default=[],
                        help="Evaluate a specific question id. Repeat for multiple ids.")
    parser.add_argument("--bm25-only", action="store_true",
                        help="Use lexical BM25 only instead of the hybrid index.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--with-llm", action="store_true",
                      help="Also call OpenAI/Gemini to evaluate generated answers.")
    mode.add_argument("--extractive", action="store_true",
                      help="Use the simple direct-search extractive baseline.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Explicitly rebuild persistent corpus embeddings before evaluation.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra diagnostic details.")
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--no-write", action="store_true",
                        help="Print summary without writing result files.")
    return parser.parse_args(argv)


def _load_saved_llm_keys() -> None:
    if not SECRETS_PATH.exists():
        return
    text = SECRETS_PATH.read_text(encoding="utf-8", errors="ignore")
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(key):
            continue
        match = re.search(rf"^\s*{key}\s*=\s*[\"']([^\"']+)[\"']\s*$",
                          text, flags=re.MULTILINE)
        if match:
            os.environ[key] = match.group(1)


def _output_args_were_explicit(raw_args: list[str]) -> bool:
    output_flags = ("--results-json", "--results-csv", "--summary-json", "--report-md")
    return any(arg == flag or arg.startswith(flag + "=") for arg in raw_args for flag in output_flags)


def _route_subset_outputs_to_tmp(args: argparse.Namespace, raw_args: list[str]) -> None:
    if args.no_write or _output_args_were_explicit(raw_args):
        return
    if args.limit is None and not args.question_id:
        return
    EVAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    args.results_json = EVAL_TMP_DIR / "rag_eval_results.json"
    args.results_csv = EVAL_TMP_DIR / "rag_eval_results.csv"
    args.summary_json = EVAL_TMP_DIR / "rag_eval_summary.json"
    args.report_md = EVAL_TMP_DIR / "rag_eval_report.md"


def _print_index_status(info: dict[str, Any], doc_count: int, chunk_count: int) -> None:
    print("[rag_eval] Index status:")
    print(f"  Index backend: {info.get('backend', 'unknown')}")
    print(f"  Index path: {info.get('index_path', '')}")
    print(f"  Index status: {info.get('index_status', 'unknown')}")
    regen = "performed" if info.get("embeddings_regenerated") else "skipped"
    print(f"  Embedding regeneration: {regen}")
    print(f"  Indexed documents: {info.get('doc_count') or doc_count}")
    print(f"  Indexed chunks: {info.get('chunk_count') or chunk_count}")
    print(f"  Embedding model: {info.get('embedding_model', '')}")
    if info.get("rebuild_performed") or info.get("rebuild_reason"):
        print(f"  Reason: {info.get('rebuild_reason')}")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot_index_mtimes(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    snapshot: dict[str, dict[str, Any]] = {}
    backup_dir = EVAL_TMP_DIR / "index-backups"
    for path in sorted(index_path.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(index_path)).replace("\\", "/")
        stat = path.stat()
        record: dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if rel == "chroma.sqlite3":
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"chroma.sqlite3.{os.getpid()}.bak"
            shutil.copy2(path, backup)
            record["sha256"] = _file_sha256(path)
            record["backup_path"] = str(backup)
        snapshot[rel] = record
    return snapshot


def _restore_read_only_index_mtimes(index_path: Path,
                                    before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not before:
        return {"restored": [], "changed": []}
    restored: list[str] = []
    changed: list[str] = []
    for rel, old in before.items():
        path = index_path / rel
        if not path.exists():
            changed.append(rel)
            continue
        stat = path.stat()
        if stat.st_mtime_ns == old["mtime_ns"]:
            backup_path = old.get("backup_path")
            if backup_path:
                Path(backup_path).unlink(missing_ok=True)
            continue
        if rel == "chroma.sqlite3" and old.get("backup_path"):
            backup = Path(str(old["backup_path"]))
            if backup.exists():
                shutil.copy2(backup, path)
                os.utime(path, ns=(path.stat().st_atime_ns, int(old["mtime_ns"])))
                backup.unlink(missing_ok=True)
                restored.append(rel)
                continue
        if stat.st_size == old["size"] and rel == "chroma.sqlite3" and old.get("sha256") == _file_sha256(path):
            os.utime(path, ns=(stat.st_atime_ns, int(old["mtime_ns"])))
            restored.append(rel)
            continue
        changed.append(rel)
    return {"restored": restored, "changed": changed}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    if args.min_pass_rate is not None and not 0.0 <= args.min_pass_rate <= 1.0:
        print("[rag_eval] --min-pass-rate must be between 0 and 1.", file=sys.stderr)
        return 2
    _route_subset_outputs_to_tmp(args, raw_args)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    eval_start = time.perf_counter()
    index_mtime_snapshot: dict[str, dict[str, Any]] = {}
    if not args.rebuild_index and not args.bm25_only:
        index_mtime_snapshot = _snapshot_index_mtimes(INDEX_DIR)
    if args.with_llm:
        _load_saved_llm_keys()

    cases = load_eval_set(args.eval_path)
    if args.question_id:
        wanted = set(args.question_id)
        cases = [case for case in cases if case.question_id in wanted]
        missing = sorted(wanted - {case.question_id for case in cases})
        if missing:
            print(f"[rag_eval] Warning: question id(s) not found: {', '.join(missing)}")
    if args.limit is not None:
        cases = cases[:args.limit]

    print(f"[rag_eval] Loaded {len(cases)} eval cases from {args.eval_path}")
    try:
        index, backend, doc_count, chunk_count, index_info = build_cli_index(
            args.bm25_only,
            rebuild_index=args.rebuild_index,
        )
    except Exception as exc:
        print(f"[rag_eval] Index load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[rag_eval] Re-run with --rebuild-index only when you want to regenerate corpus embeddings.", file=sys.stderr)
        return 2
    _print_index_status(index_info, doc_count, chunk_count)
    print(f"[rag_eval] Indexed {doc_count} docs / {chunk_count} chunks with {backend}")

    if args.with_llm:
        answerer = None
        evaluation_mode = "rag_answer"
    elif args.extractive:
        answerer = extractive_answerer
        evaluation_mode = "extractive_baseline"
    else:
        answerer = retrieval_only_answerer
        evaluation_mode = "retrieval_only"
    results = []
    for i, case in enumerate(cases, 1):
        result = evaluate_case(index, case, k=args.top_k, answerer=answerer)
        results.append(result)
        overall = result.overall_score
        overall_text = "n/a" if overall is None else f"{overall:.3f}"
        passed = case_passed(result, args.pass_threshold)
        print(
            f"[rag_eval] {i:03d}/{len(cases)} {case.question_id} "
            f"{case.question_type}: overall_score={overall_text} pass={passed}"
        )

    del index
    gc.collect()
    mtime_restore = _restore_read_only_index_mtimes(
        Path(index_info.get("index_path", INDEX_DIR)),
        index_mtime_snapshot,
    )
    extra = {
        "index_backend": backend,
        "index": index_info,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "top_k": args.top_k,
        "eval_path": str(args.eval_path),
        "evaluation_mode": evaluation_mode,
        "run_mode": evaluation_mode,
        "evaluation_timestamp": started_at,
        "evaluation_runtime_seconds": round(time.perf_counter() - eval_start, 3),
        "index_mtime_restored": mtime_restore["restored"],
        "index_mtime_changed": mtime_restore["changed"],
    }
    if args.no_write:
        summary = pass_fail_summary(
            results,
            args.pass_threshold,
            index_info=index_info,
            evaluation_mode=extra["evaluation_mode"],
        )
        summary.update(extra)
    else:
        summary = write_result_files(
            results,
            pass_threshold=args.pass_threshold,
            results_json=args.results_json,
            results_csv=args.results_csv,
            summary_json=args.summary_json,
            report_md=args.report_md,
            extra_summary=extra,
        )

    print("[rag_eval] Final summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.no_write:
        print(f"[rag_eval] Wrote {args.results_json}")
        print(f"[rag_eval] Wrote {args.results_csv}")
        print(f"[rag_eval] Wrote {args.summary_json}")
        print(f"[rag_eval] Wrote {args.report_md}")
    if args.min_pass_rate is not None:
        pass_rate = summary.get("pass_rate")
        if pass_rate is None or pass_rate < args.min_pass_rate:
            print(
                f"[rag_eval] Pass rate {_pct(pass_rate)} is below the required "
                f"{_pct(args.min_pass_rate)}.",
                file=sys.stderr,
            )
            return 1
    return 0


def _overall_score(result: RagEvalResult) -> float | None:
    metrics = result.metrics.as_dict()
    diagnostics = result.diagnostics
    if result.case.expected_unanswerable:
        return _weighted_mean([
            (diagnostics.get("abstention_accuracy"), 0.55),
            (metrics.get("answer_relevancy"), 0.25),
            (metrics.get("faithfulness"), 0.20),
        ])

    # Retrieval remains dominant (85%). Answer quality contributes 15%, so it
    # affects the result without hiding a retrieval failure.
    return _weighted_mean([
        (metrics.get("contextual_recall"), 0.30),
        (metrics.get("contextual_precision"), 0.15),
        (diagnostics.get("mrr"), 0.15),
        (diagnostics.get("evidence_hit"), 0.12),
        (diagnostics.get("exact_chunk_hit"), 0.08),
        (diagnostics.get("metadata_hit_rate"), 0.05),
        (diagnostics.get("answer_correctness"), 0.05),
        (metrics.get("answer_relevancy"), 0.03),
        (metrics.get("faithfulness"), 0.05),
        (diagnostics.get("citation_correctness"), 0.02),
    ])


def _weighted_mean(values: list[tuple[float | None, float]]) -> float | None:
    weighted = [(float(value), weight) for value, weight in values if value is not None]
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return None
    return _round(sum(value * weight for value, weight in weighted) / total_weight)


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


if __name__ == "__main__":
    raise SystemExit(main())
