"""Tests for earnings-call transcript parsing.  Run:  pytest tests/test_transcript_parsing.py -v

Covers the speaker-header regexes (dash and colon styles), the lowercase
name-particle fix ("Luca de Meo"), the false-positive guards, and the
flat-text fallback for speaker-less transcripts (free-tier data).
All tests run offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.chunker import chunk_document                    # noqa: E402
from finsight.ingest import (                                  # noqa: E402
    _SPEAKER_COLON,
    _SPEAKER_LINE,
    load_document,
)


def test_particle_names_match():
    """Names with lowercase connective particles must be recognised."""
    assert _SPEAKER_LINE.match("Luca de Meo -- Chief Executive Officer")
    assert _SPEAKER_LINE.match("Frans van Houten -- Chief Executive Officer")
    assert _SPEAKER_LINE.match("Maria della Rosa -- Chief Financial Officer")
    assert _SPEAKER_COLON.match("Luca de Meo: Thank you for the question.")


def test_plain_names_still_match():
    """The fix must not break ordinary capitalised names."""
    assert _SPEAKER_LINE.match("Tim Cook -- Chief Executive Officer")
    assert _SPEAKER_COLON.match("Tim Cook: Thank you, Suhasini.")


def test_guards_reject_prose():
    """Prose that merely resembles a speaker header must be rejected."""
    assert not _SPEAKER_LINE.match("Revenue -- up 12% year over year")
    assert not _SPEAKER_COLON.match("Note: this is a disclaimer")
    assert not _SPEAKER_COLON.match("Prepared Remarks:")
    assert not _SPEAKER_COLON.match("Thanks everyone: we did well")


def test_split_transcript_attributes_particle_speaker(tmp_path):
    """End to end: a particle-name speaker gets their own section."""
    text = (
        "Operator\n"
        "Good afternoon and welcome to the call. I will now hand over "
        "to management for prepared remarks.\n\n"
        "Luca de Meo -- Chief Executive Officer\n"
        "Thank you. Revenue this quarter grew fifteen percent driven by "
        "strong demand across all of our major product categories.\n"
    )
    f = tmp_path / "KER_TRANSCRIPT_2026-02-11.txt"
    f.write_text(text, encoding="utf-8")

    doc = load_document(f)
    titles = [s.title for s in doc.sections]
    assert "Luca de Meo (Chief Executive Officer)" in titles
    assert all(s.item == "PR" for s in doc.sections)


def test_flat_text_falls_back_to_single_section(tmp_path):
    """Speaker-less flat text (free-tier data) degrades gracefully."""
    text = (
        "Good day, and welcome to the earnings conference call. "
        "Revenue for the quarter was 90.8 billion dollars, up 5 percent "
        "year over year, with services setting an all-time record. " * 5
    )
    f = tmp_path / "AAPL_TRANSCRIPT_2026-04-30.txt"
    f.write_text(text, encoding="utf-8")

    doc = load_document(f)
    assert len(doc.sections) == 1
    assert doc.sections[0].title == "Full transcript"

    chunks = chunk_document(doc)
    assert chunks, "flat transcript must still produce chunks"
    # citation must not claim a phase the data does not contain
    assert "Prepared Remarks" not in chunks[0].citation
    assert chunks[0].citation.startswith("AAPL Earnings Call (2026-04-30)")