from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="FinSight", page_icon="📑", layout="wide")

from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.ingest import load_corpus


@st.cache_resource(show_spinner="Loading corpus...")
def boot():
    return load_corpus(FILINGS_DIR, SAMPLE_DIR)


docs = boot()
tickers = sorted({d.ticker for d in docs})

left, right = st.columns([3, 2])
with left:
    st.title("📑 FinSight")
    st.caption("LLM-augmented analysis of SEC filings")
with right:
    a, b = st.columns(2)
    a.metric("Filings", len(docs))
    b.metric("Companies", len(tickers))

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"**Companies:** {', '.join(tickers) if tickers else 'None loaded'}")
    if not docs:
        st.info("No filings found. Add .txt filings to data/filings/ or data/sample/.")
    else:
        st.success(f"{len(docs)} filing(s) loaded.")
    st.caption("COMP47250 · Project P8 · UCD Summer 2026")

st.markdown("### Loaded Filings")
if docs:
    for d in docs:
        st.write(f"- `{d.doc_id}` — {len(d.sections)} sections, "
                 f"{sum(len(s.text) for s in d.sections):,} chars")
else:
    st.info("No filings loaded yet.")
