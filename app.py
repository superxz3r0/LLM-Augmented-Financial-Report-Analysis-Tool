from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="FinSight — Filing Analysis", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.chunker import chunk_corpus
from finsight.index import build_index
from finsight.ingest import load_corpus
from finsight import rag

_SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"


def _load_saved_keys() -> None:
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        if key not in os.environ:
            try:
                val = st.secrets.get(key, "")
            except Exception:
                val = ""
            if val:
                os.environ[key] = val


def _save_keys(gemini: str, openai: str) -> None:
    _SECRETS_FILE.parent.mkdir(exist_ok=True)
    existing = _SECRETS_FILE.read_text() if _SECRETS_FILE.exists() else ""
    lines = [l for l in existing.splitlines()
             if not l.startswith("GEMINI_API_KEY") and not l.startswith("OPENAI_API_KEY")]
    if gemini:
        lines.append(f'GEMINI_API_KEY = "{gemini}"')
    if openai:
        lines.append(f'OPENAI_API_KEY = "{openai}"')
    _SECRETS_FILE.write_text("\n".join(lines) + "\n")


def _render_key_form() -> None:
    with st.form("api_key_form", clear_on_submit=False):
        gemini = st.text_input("Gemini API key", value=os.environ.get("GEMINI_API_KEY", ""),
                               type="password", placeholder="AIza…")
        openai = st.text_input("OpenAI API key (optional)",
                               value=os.environ.get("OPENAI_API_KEY", ""),
                               type="password", placeholder="sk-…")
        if st.form_submit_button("Save & apply", type="primary"):
            gemini, openai = gemini.strip(), openai.strip()
            if not gemini and not openai:
                st.error("Paste at least one key.")
            else:
                _save_keys(gemini, openai)
                if gemini:
                    os.environ["GEMINI_API_KEY"] = gemini
                if openai:
                    os.environ["OPENAI_API_KEY"] = openai
                st.success("Keys saved.")
                st.rerun()


_load_saved_keys()


def esc(text: str) -> str:
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading corpus and building index…")
def boot():
    docs = load_corpus(FILINGS_DIR, SAMPLE_DIR)
    chunks = chunk_corpus(docs)
    index, backend = build_index(chunks)
    return docs, chunks, index, backend


docs, chunks, index, backend = boot()
tickers = sorted({d.ticker for d in docs})

left, right = st.columns([3, 2])
with left:
    st.title("📑 FinSight")
    st.caption("LLM-augmented analysis of SEC filings · retrieval with citations")
with right:
    a, b, c = st.columns(3)
    a.metric("Filings", len(docs))
    b.metric("Companies", len(tickers))
    c.metric("Chunks", f"{len(chunks):,}")

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"**Companies:** {', '.join(tickers)}")
    st.write(f"**Retrieval backend:** `{backend}`")
    if not any(d.path.parent.name != "sample" for d in docs):
        st.info("Running on bundled sample data. Fetch real EDGAR filings:\n\n"
                "`python scripts/fetch_filings.py`")

    st.divider()
    st.subheader("API Keys")
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_gemini or has_openai:
        if has_gemini:
            st.success("Gemini key loaded")
        if has_openai:
            st.success("OpenAI key loaded")
        with st.expander("Update keys"):
            _render_key_form()
    else:
        st.warning("No LLM key set — running in extractive mode.")
        _render_key_form()

    st.caption("COMP47250 · Project P8 · UCD Summer 2026")

tab_qa, = st.tabs(["💬 Ask the filings"])

with tab_qa:
    top = st.columns([3, 1])
    with top[1]:
        scope = st.selectbox("Scope", ["All companies"] + tickers,
                             label_visibility="collapsed")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if not st.session_state.chat:
        st.caption("Try: *What are the main supply chain risks?* · "
                   "*How did operating expenses change?*")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(esc(msg["content"]))
            if msg.get("audit"):
                (st.success if msg["audit"]["passed"] else st.error)(msg["audit"]["summary"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for i, (cite, snippet) in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {cite}**")
                        st.caption(esc(snippet))

    if q := st.chat_input("Ask a question about the filings…"):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(esc(q))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering…"):
                history = [(m["content"], n["content"]) for m, n in
                           zip(st.session_state.chat[:-1:2], st.session_state.chat[1::2])
                           if m["role"] == "user" and n["role"] == "assistant"]
                ans = rag.answer(index, q,
                                 ticker=None if scope == "All companies" else scope,
                                 history=history)
            st.markdown(esc(ans.text))
            st.caption(f"backend: `{ans.backend}`")

            entry = {"role": "assistant", "content": ans.text,
                     "sources": [(c.citation, c.text[:300] + "…") for c, _ in ans.sources]}
            if ans.backend not in ("extractive", "none"):
                from finsight.audit import audit_answer
                report = audit_answer(ans.text, [c for c, _ in ans.sources])
                (st.success if report.passed else st.error)(report.summary())
                entry["audit"] = {"passed": report.passed, "summary": report.summary()}
            with st.expander(f"Sources ({len(ans.sources)})", expanded=True):
                for i, (c, score) in enumerate(ans.sources, 1):
                    st.markdown(f"**[{i}] {c.citation}** · relevance {score:.2f}")
                    st.caption(esc(c.text[:300]) + "…")
            st.session_state.chat.append(entry)

    if st.session_state.chat and st.button("🗑 Clear conversation"):
        st.session_state.chat = []
        st.rerun()
