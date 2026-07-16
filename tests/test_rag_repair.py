from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.chunker import Chunk


def _chunk(**kw) -> Chunk:
    defaults = dict(
        chunk_id="AAPL_10-K_2025-10-31#1",
        doc_id="AAPL_10-K_2025-10-31",
        ticker="AAPL",
        form="10-K",
        date="2025-10-31",
        item="1A",
        section_title="Risk Factors",
        text="Customer demand and sales channels could affect net sales.",
    )
    defaults.update(kw)
    return Chunk(**defaults)


class _FakeEmbeddings(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    encode_calls: list[list[str]] = []

    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        items = list(texts)
        self.encode_calls.append(items)
        return _FakeEmbeddings([[0.1, 0.2, 0.3] for _ in items])


class _FakeCollection:
    def __init__(self, count: int = 0):
        self._count = count
        self.added_ids: list[str] = []

    def count(self) -> int:
        return self._count

    def add(self, ids, embeddings, documents, metadatas):
        self.added_ids.extend(ids)
        self._count += len(ids)

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def get(self, ids, include=None):
        return {"documents": [""], "metadatas": [{}]}


class _FakeChromaClient:
    def __init__(self, collection: _FakeCollection | None = None):
        self.collection = collection
        self.deleted = 0
        self.created = 0

    def get_collection(self, name):
        if self.collection is None:
            raise RuntimeError("collection not found")
        return self.collection

    def delete_collection(self, name):
        self.deleted += 1
        self.collection = None

    def create_collection(self, name, metadata=None):
        self.created += 1
        self.collection = _FakeCollection()
        return self.collection


def _install_fake_chroma(monkeypatch, fake_client: _FakeChromaClient):
    module = types.SimpleNamespace(PersistentClient=lambda path: fake_client)
    monkeypatch.setitem(sys.modules, "chromadb", module)


def _install_fake_sentence_transformers(monkeypatch, model_cls=_FakeModel):
    module = types.SimpleNamespace(SentenceTransformer=model_cls)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _write_current_manifest(index_mod, chunks, doc_count=1, embedding_dimension=3):
    payload = index_mod._current_manifest(
        chunks,
        doc_count=doc_count,
        embedding_dimension=embedding_dimension,
    )
    index_mod._write_manifest(index_mod.index_manifest_path(), payload)
    return payload


def test_api_key_missing_and_invalid_are_explicit(monkeypatch, tmp_path):
    from finsight.llm import api_key_status, mask_api_key

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    status = api_key_status("openai", "OPENAI_API_KEY", env_path=tmp_path / ".env")
    assert not status.configured
    assert status.message == "OPENAI_API_KEY is not configured."

    monkeypatch.setenv("OPENAI_API_KEY", " x ")
    status = api_key_status("openai", "OPENAI_API_KEY", env_path=tmp_path / ".env")
    assert status.configured and not status.valid
    assert status.length == 1
    assert mask_api_key(" x ") == "<invalid or too short>"


def test_dotenv_loading_strips_quotes(monkeypatch, tmp_path):
    from finsight.llm import api_key_status

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    key = "sk-proj-12345678901234567890"
    env = tmp_path / ".env"
    env.write_text(f'OPENAI_API_KEY = " {key} "\n', encoding="utf-8")

    status = api_key_status("openai", "OPENAI_API_KEY", env_path=env)
    assert status.valid
    assert status.masked.startswith("sk-")
    assert status.masked.endswith("90")
    assert key not in status.masked


def test_openai_dependency_error_is_configuration_failure():
    from finsight.llm import openai_error

    err = openai_error(ModuleNotFoundError("No module named 'openai'"))
    assert err.category == "configuration_failure"
    assert "pip install" in str(err)


def test_metadata_filters_are_applied_before_bm25_ranking():
    from finsight.index import _BM25

    right = _chunk(text="Customer demand and indirect sales channels are important risks.")
    wrong_date = _chunk(
        chunk_id="AAPL_10-K_2024-11-01#1",
        doc_id="AAPL_10-K_2024-11-01",
        date="2024-11-01",
        text="Customer demand appears here too, but this is the wrong filing.",
    )
    wrong_item = _chunk(
        chunk_id="AAPL_10-K_2025-10-31#9",
        item="8",
        section_title="Financial Statements",
        text="Customer demand appears here too, but this is the wrong item.",
    )
    bm25 = _BM25([wrong_date, wrong_item, right])

    hits = bm25.search(
        "According to Item 1A of Apple's 2025-10-31 10-K, what about customer demand?",
        k=3,
        ticker="AAPL",
        form="10-K",
        date="2025-10-31",
        item="1A",
    )
    assert [chunk.chunk_id for chunk, _ in hits] == [right.chunk_id]
    assert bm25.search("customer demand", k=3, ticker="AAPL", form="10-Q") == []


def test_query_filter_extraction():
    from finsight.rag import extract_query_filters

    filters = extract_query_filters(
        "In Item 2 MD&A of Apple's 2026-05-01 10-Q, what changed?"
    )
    assert filters["form"] == ["10-Q"]
    assert filters["date"] == ["2026-05-01"]
    assert filters["item"] == ["2"]


def test_unanswerable_guidance_abstains_without_exact_fiscal_year(monkeypatch):
    from finsight.rag import answer

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class Index:
        def search(self, question, k, ticker=None, **filters):
            return [(_chunk(text="The filing provides revenue guidance for fiscal 2026."), 0.9)]

    ans = answer(
        Index(),
        "What revenue guidance does Apple provide for fiscal 2030 in the available filings?",
        ticker="AAPL",
        k=1,
    )
    assert "do not provide revenue guidance for fiscal 2030" in ans.text
    assert ans.sources == []


def test_numeric_currency_and_percentage_normalization():
    from finsight.rag_eval import _coverage_score

    score = _coverage_score(
        "Revenue was 1,200 million and margin was 12 percent.",
        "Revenue was $1.2 billion and margin was 12%.",
    )
    assert score >= 0.75


def test_eval_filters_are_parsed_from_question_not_gold_metadata():
    from finsight import rag_eval

    case = rag_eval.RagEvalCase(
        question_id="RAG_NO_LEAK",
        question="What changed in the disclosure?",
        gold_answer="Customer demand changed.",
        gold_sources=[
            rag_eval.GoldSource(
                ticker="AAPL", form="10-K", date="2025-10-31", item="1A",
                evidence="Customer demand changed.",
            )
        ],
    )

    assert rag_eval._case_filters(case) == {}


def test_contextual_precision_penalizes_irrelevant_chunks():
    from finsight import rag_eval

    relevant = _chunk(text="Customer demand changed materially.")
    irrelevant = _chunk(
        chunk_id="MSFT_10-Q_2024-01-01#4",
        doc_id="MSFT_10-Q_2024-01-01",
        ticker="MSFT",
        form="10-Q",
        date="2024-01-01",
        item="2",
        section_title="MD&A",
        text="Office lease expenses were recorded during the quarter.",
    )
    case = rag_eval.RagEvalCase(
        question_id="RAG_PRECISION",
        question="What changed in customer demand?",
        gold_answer="Customer demand changed materially.",
        gold_sources=[
            rag_eval.GoldSource(
                ticker="AAPL", form="10-K", date="2025-10-31", item="1A",
                evidence="Customer demand changed materially.",
                doc_id=relevant.doc_id,
                chunk_id=relevant.chunk_id,
            )
        ],
    )

    assert rag_eval._contextual_precision(
        [(relevant, 1.0), (irrelevant, 0.2)], case
    ) == 0.5


def test_comparison_correctness_requires_both_sources():
    from finsight import rag_eval

    case = rag_eval.RagEvalCase(
        question_id="RAG_COMPARE",
        question="What changed between the two filings?",
        gold_answer="Alpha revenue expanded. Beta cybersecurity incidents increased.",
        gold_sources=[
            rag_eval.GoldSource(evidence="Alpha revenue expanded."),
            rag_eval.GoldSource(evidence="Beta cybersecurity incidents increased."),
        ],
    )

    assert rag_eval._answer_correctness("Alpha revenue expanded.", case) == 0.5


def test_case_pass_requires_minimum_contextual_precision():
    from finsight import rag_eval
    from finsight.rag import RagAnswer

    case = rag_eval.RagEvalCase(
        question_id="RAG_GATE",
        question="What changed in customer demand?",
        gold_answer="Customer demand changed.",
        gold_sources=[rag_eval.GoldSource(evidence="Customer demand changed.")],
    )
    result = rag_eval.RagEvalResult(
        case=case,
        answer=RagAnswer("Customer demand changed [1].", [], "fake"),
        metrics=rag_eval.MetricScores(
            answer_relevancy=1.0,
            faithfulness=1.0,
            contextual_relevancy=1.0,
            contextual_recall=1.0,
            contextual_precision=0.19,
        ),
        diagnostics={
            "answer_correctness": 1.0,
            "citation_correctness": 1.0,
            "evidence_hit": 1.0,
            "exact_chunk_hit": 1.0,
            "mrr": 1.0,
            "metadata_hit_rate": 1.0,
            "abstention_accuracy": None,
        },
    )

    assert result.overall_score > rag_eval.DEFAULT_PASS_THRESHOLD
    assert not rag_eval.case_passed(result)


def test_answer_metrics_have_a_small_effect_on_overall_score():
    from finsight import rag_eval
    from finsight.rag import RagAnswer

    case = rag_eval.RagEvalCase(
        question_id="RAG_ANSWER_WEIGHT",
        question="What changed in customer demand?",
        gold_answer="Customer demand changed.",
        gold_sources=[rag_eval.GoldSource(evidence="Customer demand changed.")],
    )
    retrieval_metrics = dict(
        contextual_relevancy=1.0,
        contextual_recall=1.0,
        contextual_precision=1.0,
    )
    retrieval_diagnostics = {
        "evidence_hit": 1.0,
        "exact_chunk_hit": 1.0,
        "mrr": 1.0,
        "metadata_hit_rate": 1.0,
    }

    low_answer = rag_eval.RagEvalResult(
        case=case,
        answer=RagAnswer("Weak answer.", [], "fake"),
        metrics=rag_eval.MetricScores(
            answer_relevancy=0.0,
            faithfulness=0.0,
            **retrieval_metrics,
        ),
        diagnostics={
            **retrieval_diagnostics,
            "answer_correctness": 0.0,
            "citation_correctness": 0.0,
        },
    )
    strong_answer = rag_eval.RagEvalResult(
        case=case,
        answer=RagAnswer("Customer demand changed [1].", [], "fake"),
        metrics=rag_eval.MetricScores(
            answer_relevancy=1.0,
            faithfulness=1.0,
            **retrieval_metrics,
        ),
        diagnostics={
            **retrieval_diagnostics,
            "answer_correctness": 1.0,
            "citation_correctness": 1.0,
        },
    )

    assert low_answer.overall_score == 0.85
    assert strong_answer.overall_score == 1.0


def test_default_cli_mode_does_not_enable_llm():
    from finsight import rag_eval

    args = rag_eval._parse_args([])

    assert not args.with_llm
    assert not args.extractive


def test_removed_legacy_eval_fields_do_not_return():
    from finsight import rag_eval

    cases = json.loads((ROOT / "eval" / "rag_questions.json").read_text(
        encoding="utf-8"
    ))

    assert "recall_at_10" not in rag_eval.CSV_FIELDS
    assert all("review_status" not in case for case in cases)


def test_retrieval_only_answerer_does_not_call_provider(monkeypatch):
    from finsight import rag as rag_module
    from finsight import rag_eval

    monkeypatch.setenv("OPENAI_API_KEY", "configured-for-test")
    monkeypatch.setenv("GEMINI_API_KEY", "configured-for-test")
    monkeypatch.setattr(
        rag_module,
        "_openai_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retrieval-only evaluation called OpenAI")
        ),
    )
    monkeypatch.setattr(
        rag_module,
        "_gemini_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("retrieval-only evaluation called Gemini")
        ),
    )

    class Index:
        def search(self, question, k, ticker=None, **filters):
            return [(_chunk(), 0.9)]

    answer = rag_eval.retrieval_only_answerer(
        Index(), "What could affect customer demand?", 1, "AAPL", {}
    )

    assert answer.backend == "retrieval_only"
    assert answer.retrieval_sources


def test_invalid_citation_counts_against_citation_correctness():
    from finsight import rag_eval

    chunk = _chunk(text="Customer demand changed materially.")
    case = rag_eval.RagEvalCase(
        question_id="RAG_CITATION",
        question="What changed in customer demand?",
        gold_answer="Customer demand changed materially.",
        gold_sources=[
            rag_eval.GoldSource(
                evidence="Customer demand changed materially.",
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )
        ],
    )

    assert rag_eval._citation_correctness(
        "Customer demand changed [1] [99].", [(chunk, 1.0)], case
    ) == 0.5


def test_invalid_min_pass_rate_is_rejected_before_index_loading():
    from finsight import rag_eval

    assert rag_eval.main(["--min-pass-rate", "1.1", "--no-write"]) == 2


def test_single_eval_case_uses_case_metadata_filters(monkeypatch):
    from finsight import rag_eval

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    evidence = "Customer demand and indirect sales channels could materially affect net sales."
    chunk = _chunk(text=evidence)
    case = rag_eval.RagEvalCase(
        question_id="RAG_FILTER",
        question="According to Item 1A of Apple's 2025-10-31 10-K, what about customer demand?",
        gold_answer=evidence,
        ticker="AAPL",
        gold_sources=[
            rag_eval.GoldSource(
                ticker="AAPL",
                form="10-K",
                date="2025-10-31",
                item="1A",
                section="Risk Factors",
                evidence=evidence,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )
        ],
    )

    class Index:
        def search(self, question, k, ticker=None, **filters):
            assert ticker == "AAPL"
            assert filters["form"] == ["10-K"]
            assert filters["date"] == ["2025-10-31"]
            assert filters["item"] == ["1A"]
            return [(chunk, 0.9)]

    result = rag_eval.evaluate_case(Index(), case, k=1)
    assert rag_eval.case_passed(result)


def test_failure_category_is_written_to_result_row():
    from finsight import rag_eval
    from finsight.rag import RagAnswer

    case = rag_eval.RagEvalCase(
        question_id="RAG_API",
        question="What does the filing say?",
        gold_answer="Customer demand matters.",
        gold_sources=[
            rag_eval.GoldSource(
                ticker="AAPL",
                form="10-K",
                date="2025-10-31",
                item="1A",
                section="Risk Factors",
                evidence="Customer demand matters.",
            )
        ],
    )

    def answerer(index, question, k, ticker=None, filters=None):
        return RagAnswer(
            "Customer demand matters [1].",
            [(_chunk(text="Customer demand matters."), 0.9)],
            "extractive",
            "configuration_failure",
            "OPENAI_API_KEY is invalid.",
        )

    row = rag_eval.evaluate_case(None, case, k=1, answerer=answerer).to_row()
    assert row["failure_category"] == "configuration_failure"
    assert "invalid" in row["error_message"]


def test_valid_persistent_index_is_reused_without_embedding_generation(monkeypatch, tmp_path):
    import finsight.index as index_mod

    chunks = [_chunk()]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "index")
    _write_current_manifest(index_mod, chunks)
    fake_client = _FakeChromaClient(_FakeCollection(count=len(chunks)))
    _install_fake_chroma(monkeypatch, fake_client)

    class ForbiddenModel:
        def __init__(self, model_name):
            raise AssertionError("query/corpus embedding model should not load")

    _install_fake_sentence_transformers(monkeypatch, ForbiddenModel)

    index, name = index_mod.build_index(
        chunks,
        require_persistent=True,
        allow_tfidf_fallback=False,
        doc_count=1,
        load_query_encoder=False,
    )

    assert name == "hybrid(bm25+bge)"
    assert index.index_info["index_reused"] is True
    assert index.index_info["embeddings_regenerated"] is False
    assert fake_client.created == 0
    assert fake_client.deleted == 0


def test_missing_persistent_index_requires_explicit_rebuild(monkeypatch, tmp_path):
    import pytest
    import finsight.index as index_mod

    chunks = [_chunk()]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "index")
    _install_fake_chroma(monkeypatch, _FakeChromaClient(collection=None))

    with pytest.raises(index_mod.IndexLoadError, match="--rebuild-index"):
        index_mod.build_index(
            chunks,
            require_persistent=True,
            allow_tfidf_fallback=False,
            doc_count=1,
            load_query_encoder=False,
        )


def test_explicit_rebuild_generates_embeddings_and_manifest(monkeypatch, tmp_path):
    import finsight.index as index_mod

    chunks = [_chunk(), _chunk(chunk_id="AAPL_10-K_2025-10-31#2", text="Supply risk.")]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "index")
    fake_client = _FakeChromaClient(collection=None)
    _install_fake_chroma(monkeypatch, fake_client)
    _FakeModel.encode_calls = []
    _install_fake_sentence_transformers(monkeypatch, _FakeModel)

    index, _ = index_mod.build_index(
        chunks,
        rebuild_index=True,
        require_persistent=True,
        allow_tfidf_fallback=False,
        doc_count=1,
    )

    assert index.index_info["rebuild_performed"] is True
    assert index.index_info["embeddings_regenerated"] is True
    assert _FakeModel.encode_calls
    assert fake_client.created == 1
    assert index_mod.index_manifest_path().exists()


def test_changed_embedding_model_invalidates_index(monkeypatch, tmp_path):
    import pytest
    import finsight.index as index_mod

    chunks = [_chunk()]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "index")
    _write_current_manifest(index_mod, chunks)
    monkeypatch.setattr(index_mod.settings, "embedding_model", "different-model")
    _install_fake_chroma(monkeypatch, _FakeChromaClient(_FakeCollection(count=len(chunks))))

    with pytest.raises(index_mod.IndexLoadError, match="embedding_model changed"):
        index_mod.build_index(
            chunks,
            require_persistent=True,
            allow_tfidf_fallback=False,
            doc_count=1,
            load_query_encoder=False,
        )


def test_changed_chunk_configuration_invalidates_index(monkeypatch, tmp_path):
    import pytest
    import finsight.index as index_mod

    chunks = [_chunk()]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "index")
    _write_current_manifest(index_mod, chunks)
    monkeypatch.setattr(index_mod.settings, "chunk_size", index_mod.settings.chunk_size + 1)
    _install_fake_chroma(monkeypatch, _FakeChromaClient(_FakeCollection(count=len(chunks))))

    with pytest.raises(index_mod.IndexLoadError, match="chunk_size changed"):
        index_mod.build_index(
            chunks,
            require_persistent=True,
            allow_tfidf_fallback=False,
            doc_count=1,
            load_query_encoder=False,
        )


def test_repeated_reuse_does_not_modify_manifest_and_is_cwd_independent(monkeypatch, tmp_path):
    import os
    import finsight.index as index_mod

    chunks = [_chunk()]
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path / "abs-index")
    _write_current_manifest(index_mod, chunks)
    manifest = index_mod.index_manifest_path()
    before_mtime = manifest.stat().st_mtime_ns
    fake_client = _FakeChromaClient(_FakeCollection(count=len(chunks)))
    _install_fake_chroma(monkeypatch, fake_client)
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    for _ in range(2):
        index, _ = index_mod.build_index(
            chunks,
            require_persistent=True,
            allow_tfidf_fallback=False,
            doc_count=1,
            load_query_encoder=False,
        )
        assert index.index_info["index_reused"] is True

    assert os.getcwd() == str(other_cwd)
    assert manifest.stat().st_mtime_ns == before_mtime
    assert fake_client.created == 0
    assert fake_client.deleted == 0


def test_rag_eval_writes_consistent_markdown_csv_and_summary(tmp_path):
    from finsight import rag_eval
    from finsight.rag import RagAnswer

    chunk = _chunk(text="Customer demand matters.")
    case = rag_eval.RagEvalCase(
        question_id="RAG_REPORT",
        question="What matters?",
        question_type="unit",
        ticker="AAPL",
        gold_answer="Customer demand matters.",
        gold_sources=[
            rag_eval.GoldSource(
                ticker="AAPL",
                form="10-K",
                date="2025-10-31",
                item="1A",
                evidence="Customer demand matters.",
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )
        ],
    )

    def answerer(index, question, k, ticker=None, filters=None):
        return RagAnswer("Customer demand matters [1].", [(chunk, 1.0)], "fake")

    result = rag_eval.evaluate_case(None, case, k=1, answerer=answerer)
    paths = {
        "results_json": tmp_path / "rag_eval_results.json",
        "results_csv": tmp_path / "rag_eval_results.csv",
        "summary_json": tmp_path / "rag_eval_summary.json",
        "report_md": tmp_path / "rag_eval_report.md",
    }
    summary = rag_eval.write_result_files(
        [result],
        **paths,
        extra_summary={
            "index_backend": "hybrid(bm25+bge)",
            "index": {
                "backend": "chroma",
                "index_path": str(tmp_path / "index"),
                "index_status": "existing index loaded",
                "index_reused": True,
                "embeddings_regenerated": False,
                "rebuild_requested": False,
                "rebuild_performed": False,
                "rebuild_reason": "",
                "manifest_version": 1,
                "embedding_model": "fake",
                "embedding_dimension": 3,
                "chunk_size": 900,
                "chunk_overlap": 150,
                "load_time_seconds": 0.01,
            },
            "evaluation_mode": "unit",
            "evaluation_timestamp": "2026-07-13T00:00:00+00:00",
            "evaluation_runtime_seconds": 0.02,
        },
    )

    rows = list(csv.DictReader(paths["results_csv"].open(encoding="utf-8-sig")))
    summary_json = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    report = paths["report_md"].read_text(encoding="utf-8")
    assert summary["total"] == 1
    assert len(rows) == summary_json["total"] == 1
    assert rows[0]["question_id"] == "RAG_REPORT"
    assert "Existing index reused | Yes" in report
    assert "Actual answer backend(s) | fake: 1" in report
    assert "Pass rate:" in report
    assert "Recall@10" not in report
