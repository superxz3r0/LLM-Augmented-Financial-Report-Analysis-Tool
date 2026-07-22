from pathlib import Path

import pytest

from finsight import diff
from finsight.ingest import Document, Section


def _doc(
    ticker: str,
    form: str,
    date: str,
    sections: dict[str, str],
    *,
    suffix: str = "",
) -> Document:
    return Document(
        ticker=ticker,
        form=form,
        date=date,
        path=Path(f"{ticker}_{form}_{date}{suffix}.txt"),
        sections=[
            Section(item=item, title=f"Item {item}", text=text)
            for item, text in sections.items()
        ],
    )


def test_pairing_uses_previous_same_ticker_and_form_regardless_of_input_order():
    k23 = _doc("AAA", "10-K", "2023-01-01", {"1": "alpha"})
    k24 = _doc("AAA", "10-K", "2024-01-01", {"1": "beta"})
    q24 = _doc("AAA", "10-Q", "2024-01-01", {"1": "gamma"})
    other = _doc("BBB", "10-K", "2024-01-01", {"1": "delta"})

    pairs = diff.pair_documents_with_previous([k24, other, q24, k23])
    by_id = {pair.current.doc_id: pair.previous for pair in pairs}

    assert by_id[k23.doc_id] is None
    assert by_id[k24.doc_id] is k23
    assert by_id[q24.doc_id] is None
    assert by_id[other.doc_id] is None


def test_same_date_duplicate_is_not_used_as_a_predecessor():
    older = _doc("AAA", "10-K", "2022-01-01", {"1": "old"})
    first = _doc("AAA", "10-K", "2023-01-01", {"1": "first"}, suffix="-a")
    second = _doc("AAA", "10-K", "2023-01-01", {"1": "second"}, suffix="-b")

    pairs = diff.pair_documents_with_previous([second, first, older])
    current_date_pairs = [p for p in pairs if p.current.date == "2023-01-01"]

    assert len(current_date_pairs) == 2
    assert all(pair.previous is older for pair in current_date_pairs)


def test_identical_disclosure_has_zero_change_and_added_section_is_normalized():
    old = _doc("AAA", "10-K", "2023-01-01", {"1": "alpha beta gamma delta"})
    same = _doc("AAA", "10-K", "2024-01-01", {"1": "ALPHA beta gamma delta"})
    added = _doc(
        "AAA",
        "10-K",
        "2025-01-01",
        {"1": "alpha beta gamma delta", "1A": "risk cyber export control"},
    )

    signals = diff.compute_disclosure_signals([added, old, same])
    by_id = {signal.doc_id: signal for signal in signals}

    assert by_id[old.doc_id].score is None
    assert by_id[old.doc_id].reason == "no_predecessor"
    assert by_id[same.doc_id].score == 0.0
    assert by_id[added.doc_id].score == pytest.approx(4 / 12)
    assert by_id[added.doc_id].new_removed_token_mass == 4
    assert by_id[added.doc_id].sections_added == 1


@pytest.mark.parametrize(
    ("new_text", "expected", "category"),
    [
        # Dice=.50: substantive changed mass=10 over comparison mass=20.
        ("a b c d e k l m n o", 0.5, "substantive_token_mass"),
        # Dice=.80: minor changed mass=4, downweighted by .25 => 1/20.
        ("a b c d e f g h k l", 0.05, "minor_token_mass"),
        # Dice=.90 is boilerplate-equivalent and contributes no signal.
        ("a b c d e f g h i k", 0.0, None),
        # No overlap is a replacement/new-removed change.
        ("k l m n o p q r s t", 1.0, "new_removed_token_mass"),
    ],
)
def test_signal_uses_diff_categories_and_documented_weights(
    new_text: str, expected: float, category: str | None
):
    old = _doc("AAA", "10-K", "2023-01-01", {"1": "a b c d e f g h i j"})
    new = _doc("AAA", "10-K", "2024-01-01", {"1": new_text})

    signal = diff.compute_disclosure_signals([new, old])[1]

    assert signal.score == pytest.approx(expected)
    assert 0.0 <= signal.score <= 1.0
    if category is not None:
        assert getattr(signal, category) > 0


def test_transcripts_and_empty_comparisons_are_explicitly_unavailable():
    t1 = _doc("AAA", "TRANSCRIPT", "2023-01-01", {"PR": "prepared remarks"})
    t2 = _doc("AAA", "TRANSCRIPT", "2024-01-01", {"QA": "questions"})
    empty1 = _doc("BBB", "10-K", "2023-01-01", {"1": "!!!"})
    empty2 = _doc("BBB", "10-K", "2024-01-01", {"1": "..."})

    by_id = {
        signal.doc_id: signal
        for signal in diff.compute_disclosure_signals([t2, empty2, t1, empty1])
    }

    assert by_id[t1.doc_id].score is None
    assert by_id[t2.doc_id].score is None
    assert by_id[t2.doc_id].reason == "unsupported_transcript"
    assert by_id[empty2.doc_id].score is None
    assert by_id[empty2.doc_id].reason == "empty_comparison"
    assert by_id[empty2.doc_id].predecessor_doc_id == empty1.doc_id


def test_whole_corpus_path_is_deterministic_and_profiles_each_filing_once(monkeypatch):
    docs = [
        _doc("AAA", "10-K", "2022-01-01", {"1": "alpha beta"}),
        _doc("AAA", "10-K", "2023-01-01", {"1": "alpha gamma"}),
        _doc("AAA", "10-K", "2024-01-01", {"1": "alpha delta"}),
    ]
    original = diff._profile_document
    calls: list[str] = []

    def counted(document: Document):
        calls.append(document.doc_id)
        return original(document)

    monkeypatch.setattr(diff, "_profile_document", counted)
    first = diff.compute_disclosure_signals(reversed(docs))
    assert len(calls) == len(docs)

    monkeypatch.setattr(diff, "_profile_document", original)
    second = diff.compute_disclosure_signals(docs)

    assert first == second
    assert diff.disclosure_signal_map(docs) == {
        signal.doc_id: signal.score for signal in second
    }

