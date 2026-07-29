from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight import audit, diff, extract
from finsight.chunker import chunk_corpus, chunk_document
from finsight.config import SAMPLE_DIR
from finsight.ingest import clean_text, load_corpus
from finsight.returns import ols


@pytest.fixture(scope="session")
def docs():
    d = load_corpus(SAMPLE_DIR)
    assert d, "sample corpus must load"
    return d


def test_sections_split_on_items(docs):
    aurb = next(d for d in docs if "AURB_10-K_2024" in d.doc_id)
    items = {s.item for s in aurb.sections}
    assert "1A" in items and "7" in items


def test_clean_text_strips_html():
    html = "<html><body><p>Item 1A. Risk&nbsp;Factors</p><script>x=1</script><div>Body text here.</div></body></html>"
    out = clean_text(html)
    assert "<" not in out and "script" not in out
    assert "Risk Factors" in out and "Body text here." in out


def test_toc_lines_are_skipped():
    text = ("Item 1A. Risk Factors\nItem 7. MD&A\n\n"
            "Item 1A. Risk Factors\n" + "Real risk disclosure. " * 30 + "\n\n"
            "Item 7. MD&A\n" + "Real discussion of results. " * 30)
    from finsight.ingest import _split_sections
    sections = _split_sections(text)
    assert len(sections) == 2
    assert all(len(s.text) > 200 for s in sections)


def test_bare_heading_recovers_section_never_prefixed_with_item_n():
    """Regression for a real JPMorgan 10-K: its table of contents states
    'Item 7.' once and the body never repeats it, captioning the section
    with a bare 'Management's discussion and analysis' heading instead. The
    old code let Item 7's real content get swallowed into whichever item
    happened to be last in the TOC (Item 15), because ITEM_PATTERN only ever
    matched inside the TOC block and its last match's body ran unbounded to
    EOF. The bare-heading fallback should recover Item 7 as its own section
    and cap Item 15's TOC-only body well short of the whole document."""
    toc = "\n".join(f"Item {n}. Some Item {n} Title ... {n}" for n in
                     ["1", "1A", "7", "7A", "8", "15"])
    body = ("Cover page and signature boilerplate. " * 20 + "\n\n"
            "Management's discussion and analysis:\n" +
            "Real MD&A content discussing results of operations. " * 40)
    text = toc + "\n\n" + body
    from finsight.ingest import _split_sections
    sections = {s.item: s for s in _split_sections(text)}
    assert "7" in sections
    assert "results of operations" in sections["7"].text.lower()
    assert "15" not in sections or len(sections["15"].text) < len(sections["7"].text)


def test_chunks_carry_citation_metadata(docs):
    chunks = chunk_document(docs[0])
    assert chunks
    c = chunks[0]
    assert c.ticker and c.form and c.date and c.item
    assert c.ticker in c.citation and c.date in c.citation


def test_chunk_sizes_bounded(docs):
    from finsight.config import settings
    for c in chunk_corpus(docs):
        assert len(c.text) <= settings.chunk_size + settings.chunk_overlap + 200


def test_unsplittable_table_is_still_bounded():
    """Real SEC financial tables have no '.', '!' or '?' inside them, so
    _SENT_SPLIT can't break them up — a multi-row table bigger than
    chunk_size must still be hard-wrapped instead of passing through as one
    oversized chunk (regression: a real Apple 10-K segment table produced a
    2,532-char chunk against a 900-char target before this was fixed)."""
    from finsight.config import settings
    from finsight.ingest import Document, Section
    row = " ".join(f"Segment{i} $ {i * 1111},000 {i}" for i in range(40))
    table = "The following table shows net sales by segment:\n\n" + row + "\n" * 1 + row
    doc = Document(ticker="X", form="10-K", date="2025-01-01", path=Path("x"),
                    sections=[Section(item="7", title="MD&A", text=table)])
    cap = settings.chunk_size + settings.chunk_overlap + 200
    for c in chunk_document(doc):
        assert len(c.text) <= cap


def test_extract_finds_guidance_and_risks(docs):
    aurb24 = next(d for d in docs if "AURB_10-K_2024" in d.doc_id)
    sig = extract.extract_signals(aurb24)
    assert sig.risk_factor_count >= 3
    assert any("2.7" in g for g in sig.revenue_guidance)
    assert any("310" in g for g in sig.capex_guidance)


def test_risk_count_handles_real_filing_text_layouts():
    body = "The explanatory discussion describes the business impact and mitigation. " * 6

    # Some filings keep bold labels as short prefixes in a larger paragraph.
    inline = "\n\n".join(
        f"Risk area {i}. {body}" for i in range(10)
    )
    assert extract._count_risk_factors(inline) == 10

    # Other text conversions remove the separator between a bold heading and
    # its body. The capitalised subject still exposes the structural boundary.
    glued = "\n\n".join(
        f"Supply disruption could harm operations {body}" for _ in range(8)
    )
    assert extract._count_risk_factors(glued) == 8

    # Title-cased templates need no explicit risk keyword in every heading.
    title_case = "\n\n".join(
        part
        for i in range(15)
        for part in (f"International Market Conditions Number {i}", body)
    )
    assert extract._count_risk_factors(title_case) == 15


def test_sentiment_direction_flips_between_years(docs):
    a24 = next(d for d in docs if "AURB_10-K_2024" in d.doc_id)
    a25 = next(d for d in docs if "AURB_10-K_2025" in d.doc_id)
    s24 = extract.extract_signals(a24).sentiment_score
    s25 = extract.extract_signals(a25).sentiment_score
    assert s24 > s25


def test_diff_surfaces_new_and_removed_disclosures(docs):
    a24 = next(d for d in docs if "AURB_10-K_2024" in d.doc_id)
    a25 = next(d for d in docs if "AURB_10-K_2025" in d.doc_id)
    items = diff.diff_documents(a24, a25)
    kinds = {i.kind for i in items}
    assert "new" in kinds and "removed" in kinds
    new_text = " ".join(i.new_text for i in items if i.kind == "new")
    assert "export control" in new_text.lower()


def test_diff_identical_docs_is_quiet(docs):
    a24 = next(d for d in docs if "AURB_10-K_2024" in d.doc_id)
    items = diff.diff_documents(a24, a24)
    assert not [i for i in items if i.kind in ("new", "substantive", "removed")]


class _FakeChunk:
    def __init__(self, text):
        self.text = text


def test_audit_passes_grounded_answer():
    src = [_FakeChunk("Total revenue grew 18 percent to $2.4 billion in fiscal 2024, "
                      "driven by record demand for collaborative robots.")]
    ans = "Revenue grew 18 percent to $2.4 billion [1], driven by demand for collaborative robots [1]."
    report = audit.audit_answer(ans, src)
    assert report.passed, report.summary()


def test_audit_flags_fabricated_number():
    src = [_FakeChunk("Total revenue grew 18 percent to $2.4 billion in fiscal 2024.")]
    ans = "Revenue grew 42 percent to $9.9 billion."
    report = audit.audit_answer(ans, src)
    assert not report.passed
    assert any(s.unsupported_numbers for s in report.sentences)


def test_ols_recovers_known_slope():
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(scale=0.1, size=200)
    r = ols(x, y, window=5)
    assert abs(r.coef - 0.5) < 0.05
    assert r.t_stat > 10 and r.r2 > 0.8


def test_ols_with_control():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    mkt = rng.normal(size=300)
    y = 0.3 * x + 0.8 * mkt + rng.normal(scale=0.1, size=300)
    r = ols(x, y, window=20, controls={"mkt": mkt})
    assert abs(r.coef - 0.3) < 0.05
    assert abs(r.controls["mkt"][0] - 0.8) < 0.05


def test_chunks_never_start_mid_word(docs):
    for c in chunk_corpus(docs):
        first = c.text.split(" ", 1)[0]
        assert first[0].isupper() or first[0].isdigit() or first[0] in "\"'($•—", \
            f"chunk starts mid-word: {c.text[:60]!r}"


def test_retrieval_dedupes_boilerplate(docs):
    from finsight.rag import _dedupe

    class FakeChunk:
        def __init__(self, text): self.text = text

    boiler = ("Global supply chains can be highly concentrated and geopolitical "
              "tensions or conflict could result in significant disruptions to output.")
    hits = [
        (FakeChunk(boiler), 0.95),
        (FakeChunk(boiler + " "), 0.94),
        (FakeChunk(boiler.replace("output", "production")), 0.93),
        (FakeChunk("Completely different passage about climate risk and weather."), 0.90),
        (FakeChunk("Another distinct passage about outsourcing partner prepayments."), 0.88),
    ]
    kept = _dedupe(hits, k=5)
    texts = [c.text for c, _ in kept]
    assert len(texts) == 3
    assert texts[0] == boiler


def _fund(**kw):
    from finsight.metrics import Fundamentals
    return Fundamentals(period=kw.pop("period", "FY"), **kw)


def test_ratio_formulas():
    from finsight.metrics import compute_ratios
    r = compute_ratios(_fund(revenue=1000, cost_of_revenue=600, operating_income=200,
                             net_income=150, total_assets=2000, total_equity=1000,
                             total_debt=500, current_assets=800, current_liabilities=400,
                             operating_cash_flow=250, capex=100, interest_expense=20))
    assert abs(r["gross_margin"] - 0.4) < 1e-9
    assert abs(r["operating_margin"] - 0.2) < 1e-9
    assert abs(r["current_ratio"] - 2.0) < 1e-9
    assert abs(r["debt_to_equity"] - 0.5) < 1e-9
    assert abs(r["roe"] - 0.15) < 1e-9
    assert abs(r["fcf"] - 150) < 1e-9
    assert abs(r["interest_coverage"] - 10.0) < 1e-9


def test_ratios_never_fake_zeroes_on_missing_data():
    from finsight.metrics import compute_ratios
    r = compute_ratios(_fund(revenue=1000))
    assert r["net_margin"] is None and r["roe"] is None and r["fcf"] is None


def test_growth_analysis_and_negative_base():
    from finsight.metrics import growth_analysis
    g = growth_analysis([_fund(period="A", revenue=100, net_income=-5),
                         _fund(period="B", revenue=120, net_income=10)])
    assert abs(g["revenue"][0] - 0.2) < 1e-9
    assert g["net_income"][0] is None


def test_health_check_fires_expected_flags():
    from finsight.metrics import health_check
    weak = [
        _fund(period="FY1", revenue=1000, operating_income=150, net_income=100,
              current_assets=400, current_liabilities=500, total_debt=2500,
              total_equity=1000, operating_cash_flow=-50, capex=30, interest_expense=120),
    ]
    flags = health_check(weak)
    fired = {f.metric: f.severity for f in flags}
    assert fired.get("current_ratio") == "red"
    assert fired.get("debt_to_equity") == "red"
    assert fired.get("fcf") == "red"
    assert "interest_coverage" in fired


def test_sample_fundamentals_load_and_flag():
    import json
    from finsight.config import SAMPLE_DIR
    from finsight.metrics import Fundamentals, health_check, growth_analysis
    payload = json.loads((SAMPLE_DIR / "AURB_fundamentals.json").read_text())
    periods = [Fundamentals(**p) for p in payload["periods"]]
    g = growth_analysis(periods)
    assert g["revenue"][0] is not None and g["revenue"][0] > 0.15
    flags = health_check(periods)
    assert any(f.metric == "fcf" and f.severity == "green" for f in flags)


def test_bm25_ranks_exact_terms_first(docs):
    from finsight.index import _BM25
    from finsight.chunker import chunk_corpus
    bm = _BM25(chunk_corpus(docs))
    hits = bm.search("export control regulations autonomous", k=3)
    assert hits and "export control" in hits[0][0].text.lower()


def test_rrf_fusion_merges_rankers(docs):
    from finsight.index import build_index
    from finsight.chunker import chunk_corpus
    index, name = build_index(chunk_corpus(docs))
    assert name.startswith("hybrid(bm25+")
    hits = index.search("supply chain risk", k=5)
    assert len(hits) == 5
    assert len({c.chunk_id for c, _ in hits}) == 5


def _facts_fixture():
    def entry(fy, val, filed, form="10-K", fp="FY"):
        return {"fy": fy, "val": val, "filed": filed, "form": form, "fp": fp}
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            entry(2023, 1000, "2023-11-01"),
            entry(2024, 1200, "2024-11-01"),
            entry(2023, 1010, "2024-11-01"),
            entry(2024, 999, "2024-02-01", form="10-Q", fp="Q1"),
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            entry(2023, 100, "2023-11-01"), entry(2024, 150, "2024-11-01")]}},
        "Assets": {"units": {"USD": [
            entry(2023, 5000, "2023-11-01"), entry(2024, 5500, "2024-11-01")]}},
    }}}


def test_xbrl_parser_picks_annual_and_latest_filed():
    from finsight.xbrl import parse_companyfacts
    periods = parse_companyfacts(_facts_fixture())
    assert [p.period for p in periods] == ["FY2023", "FY2024"]
    assert periods[0].revenue == 1010
    assert periods[1].revenue == 1200
    assert periods[1].net_income == 150
    assert periods[1].capex is None


def test_llm_payload_validation_rejects_garbage():
    from finsight.extract import _validate_llm_payload
    ok = _validate_llm_payload('{"revenue_guidance": ["$2.7 billion"], "capex_guidance": []}')
    assert ok and ok["revenue_guidance"] == ["$2.7 billion"]
    assert _validate_llm_payload("not json") is None
    assert _validate_llm_payload('{"revenue_guidance": "oops", "capex_guidance": []}') is None
    assert _validate_llm_payload('{"revenue_guidance": [1, 2], "capex_guidance": []}') is None
    fenced = '```json\n{"revenue_guidance": [], "capex_guidance": ["$240 million"]}\n```'
    assert _validate_llm_payload(fenced)["capex_guidance"] == ["$240 million"]


def test_sqlite_signal_cache_roundtrip(docs):
    from finsight import store
    store.put_signals("FAKE_DOC", {"sentiment_score": 0.5})
    assert store.get_signals("FAKE_DOC")["sentiment_score"] == 0.5
    d = docs[0]
    first = store.cached_extract(d, use_llm=False)
    second = store.cached_extract(d, use_llm=False)
    assert first == second and "risk_factor_count" in first
