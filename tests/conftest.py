"""Test-wide fixtures.

Isolates the Chroma vector store the test suite builds from the one the
real app uses. Both finsight.index._ChromaIndex and any live `streamlit run
app.py` process persist to the same on-disk path (DATA_DIR/index) under the
same hardcoded collection name ("filings"). Without this fixture, running
pytest while the app is mid-build against the real corpus races: a test's
build_index() call sees a fingerprint mismatch, deletes the "filings"
collection to rebuild it for the tiny sample corpus, and evicts the app's
in-progress collection out from under it — the app's later `.add()` then
raises `Collection [...] does not exist` and silently falls back to the far
weaker TF-IDF backend. Point tests at a throwaway directory instead so they
can never collide with (or corrupt) the real persisted index.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_index_dir(tmp_path_factory):
    import finsight.index as index_mod

    index_mod.INDEX_DIR = tmp_path_factory.mktemp("chroma_index")
    yield
