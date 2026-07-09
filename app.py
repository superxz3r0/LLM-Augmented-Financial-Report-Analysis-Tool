from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="FinSight — Filing Analysis", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

from finsight import diff as diff_mod
from finsight import rag, sentiment
from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.chunker import chunk_corpus
from finsight.index import build_index
from finsight.ingest import load_corpus

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
        gemini = st.text_input(
            "Gemini API key", value=os.environ.get("GEMINI_API_KEY", ""),
            type="password", placeholder="AIza…",
            help="Free tier via Google AI Studio — gemini-2.5-flash",
        )
        openai = st.text_input(
            "OpenAI API key (optional)", value=os.environ.get("OPENAI_API_KEY", ""),
            type="password", placeholder="sk-…",
        )
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
                st.success("Keys saved — will reload automatically on next start.")
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
n_real = len([d for d in docs if d.path.parent.name != "sample"])

left, right = st.columns([3, 2])
with left:
    st.title("📑 FinSight")
    st.caption("LLM-augmented analysis of SEC filings · retrieval with citations, "
               "computed fundamentals, and substantive-change detection")
with right:
    a, b, c = st.columns(3)
    a.metric("Filings", len(docs))
    b.metric("Companies", len(tickers))
    c.metric("Chunks", f"{len(chunks):,}")

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"**Companies:** {', '.join(tickers)}")
    st.write(f"**Retrieval backend:** `{backend}`")
    if n_real == 0:
        st.info("Running on bundled synthetic samples. Add real EDGAR filings:\n\n"
                "`python scripts/fetch_filings.py`")
    else:
        st.success(f"{n_real} real EDGAR filings loaded.")
    with st.expander("How it works"):
        st.markdown(
            "1. **Ingest** — Item-level section split of 10-K/10-Q text\n"
            "2. **Index** — sentence-aware chunks → vector store\n"
            "3. **Answer** — RAG with mandatory `[n]` citations + hallucination audit\n"
            "4. **Compute** — signals, ratios, health flags\n"
            "5. **Compare** — substantive-change diff between filings\n"
            "6. **Correlate** — signals vs forward returns (OLS)")

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
        st.warning("No LLM key set — running in extractive mode. "
                   "Paste a key below to enable generated answers.")
        _render_key_form()

    st.caption("COMP47250 · Project P8 · UCD Summer 2026")

tab_qa, tab_signals, tab_fund, tab_diff, tab_returns = st.tabs(
    ["💬 Ask the filings", "📊 Structured signals", "📐 Fundamentals",
     "🔍 Filing diff", "📈 Signal → returns"]
)

with tab_qa:
    top = st.columns([3, 1])
    with top[1]:
        scope = st.selectbox("Scope", ["All companies"] + tickers,
                             label_visibility="collapsed",
                             help="Restrict retrieval to one company")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if not st.session_state.chat:
        st.caption("Try: *What are the main supply chain risks?* · "
                   "*How did operating expenses change?* · "
                   "*What does management say about AI?*")

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

with tab_signals:
    st.caption("Rule-based extraction of four signal types per filing: revenue "
               "guidance, capex guidance, risk-factor count, FinBERT sentiment.")
    c1, c2 = st.columns(2)
    t_sel = c1.selectbox("Company", tickers, key="sig_ticker")
    t_docs = sorted([d for d in docs if d.ticker == t_sel], key=lambda d: d.date, reverse=True)
    d_sel = c2.selectbox("Filing", t_docs, format_func=lambda d: f"{d.form} · {d.date}")

    from finsight.store import cached_extract
    with st.spinner("Extracting signals (cached after first run)…"):
        sig_d = cached_extract(d_sel)
    from finsight.extract import SignalSet
    sig = SignalSet(**sig_d)
    a, b, c, d4 = st.columns(4)
    a.metric("Sentiment (MD&A)", f"{sig.sentiment_score:+.3f}",
             help=f"backend: {sig.sentiment_backend}")
    st.caption(f"extraction: `{sig.method}`" +
               (f" · regex cross-check {'agrees ✓' if sig.xcheck_agree else 'DISAGREES ✗'}"
                if sig.xcheck_agree is not None else ""))
    b.metric("Risk factors", sig.risk_factor_count)
    c.metric("Revenue guidance hits", len(sig.revenue_guidance))
    d4.metric("Capex guidance hits", len(sig.capex_guidance))
    if sig.revenue_guidance:
        st.markdown("**Revenue guidance:** " + esc(" · ".join(sig.revenue_guidance)))
    if sig.capex_guidance:
        st.markdown("**Capex guidance:** " + esc(" · ".join(sig.capex_guidance)))

    st.divider()
    st.markdown("**Sentiment over time** — MD&A sentiment per filing, per company.")
    if st.button("Compute sentiment trend across the corpus"):
        import pandas as pd
        rows = []
        prog = st.progress(0.0)
        for i, d in enumerate(docs):
            s = cached_extract(d, use_llm=False)
            rows.append({"date": d.date, "ticker": d.ticker, "sentiment": s["sentiment_score"]})
            prog.progress((i + 1) / len(docs))
        prog.empty()
        df = pd.DataFrame(rows).pivot_table(index="date", columns="ticker", values="sentiment")
        st.line_chart(df)

with tab_fund:
    from finsight import metrics as metrics_mod

    st.caption("Computed financial analysis: ratios, YoY growth, and rule-based health flags.")
    source = st.radio("Data source",
                      ["Sample (offline)", "SEC XBRL (structured)", "yfinance"],
                      horizontal=True, label_visibility="collapsed")
    periods = []
    if source == "Sample (offline)":
        payload = json.loads((SAMPLE_DIR / "AURB_fundamentals.json").read_text())
        periods = [metrics_mod.Fundamentals(**{k: v for k, v in p.items()})
                   for p in payload["periods"]]
        st.caption(f"Synthetic fundamentals for {payload['ticker']} (USD millions)")
    else:
        cc1, cc2 = st.columns([1, 3])
        live_ticker = cc1.text_input("Ticker", value="AAPL", label_visibility="collapsed")
        if cc2.button("Fetch statements"):
            with st.spinner("Fetching…"):
                try:
                    if source.startswith("SEC"):
                        from finsight.xbrl import fetch_fundamentals
                        st.session_state.fund = fetch_fundamentals(live_ticker)
                        st.caption("Source: SEC XBRL companyfacts — issuer-filed structured data.")
                    else:
                        st.session_state.fund = metrics_mod.from_yfinance(live_ticker)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        periods = st.session_state.get("fund", [])

    if periods:
        import pandas as pd

        ratio_rows = {p.period: metrics_mod.compute_ratios(p) for p in periods}
        df = pd.DataFrame(ratio_rows)
        pct = ["gross_margin", "operating_margin", "net_margin", "roe", "roa", "fcf_margin"]

        def _fmt(idx, v):
            if v is None or (isinstance(v, float) and v != v):
                return "—"
            if idx in pct:
                return f"{v:.1%}"
            return f"{v:,.0f}" if idx == "fcf" else f"{v:.2f}"

        styled = pd.DataFrame(
            {col: [_fmt(idx, df.at[idx, col]) for idx in df.index] for col in df.columns},
            index=df.index, dtype="object",
        )
        st.dataframe(styled, use_container_width=True)

        g = metrics_mod.growth_analysis(periods)
        if any(any(v is not None for v in vals) for vals in g.values()):
            st.markdown("**YoY growth**")
            glabels = [f"{a.period}→{b.period}" for a, b in zip(periods, periods[1:])]
            gdf = pd.DataFrame(g, index=glabels).T
            st.dataframe(gdf.map(lambda v: "—" if v is None else f"{v:+.1%}"),
                         use_container_width=True)

        st.markdown("**Health check**")
        icon = {"red": "🔴", "amber": "🟡", "green": "🟢"}
        for f in metrics_mod.health_check(periods):
            st.markdown(f"{icon[f.severity]} **{f.metric}** — {f.message}  \n"
                        f"<small>rule: `{f.rule}`</small>", unsafe_allow_html=True)
        st.caption("Screening heuristics for further investigation — not investment advice.")

with tab_diff:
    st.caption("Detects substantive disclosure changes between two filings of the same company, "
               "filtering out boilerplate and numeric-only edits.")
    by_ticker = {t: sorted([d for d in docs if d.ticker == t], key=lambda d: d.date)
                 for t in tickers}
    eligible = [t for t, ds in by_ticker.items() if len(ds) >= 2]
    if not eligible:
        st.info("Need at least two filings for one company — fetch more with "
                "`scripts/fetch_filings.py`.")
    else:
        c1, c2, c3 = st.columns(3)
        t = c1.selectbox("Company", eligible, key="diff_ticker")
        ds = by_ticker[t]
        old_doc = c2.selectbox("Older filing", ds[:-1],
                               format_func=lambda d: f"{d.form} · {d.date}")
        newer = [d for d in ds if d.date > old_doc.date]
        new_doc = c3.selectbox("Newer filing", newer, index=len(newer) - 1,
                               format_func=lambda d: f"{d.form} · {d.date}")
        if st.button("Run diff", type="primary"):
            with st.spinner("Comparing filings…"):
                items = diff_mod.diff_documents(old_doc, new_doc)
            shown = [i for i in items if i.kind in ("new", "substantive", "removed")]
            st.success(f"{len(shown)} substantive changes "
                       f"({len(items) - len(shown)} minor edits suppressed)")
            badge = {"new": "🟢 NEW", "substantive": "🟠 CHANGED", "removed": "🔴 REMOVED"}
            for it in shown:
                st.markdown(f"**{badge[it.kind]}** · Item {it.item} · "
                            f"similarity {it.similarity}")
                lcol, rcol = st.columns(2)
                lcol.caption(f"{old_doc.form} · {old_doc.date}")
                lcol.text(it.old_text or "—")
                rcol.caption(f"{new_doc.form} · {new_doc.date}")
                rcol.text(it.new_text or "—")
                st.divider()

with tab_returns:
    st.caption("Regresses filing sentiment against 5- and 20-day forward returns, "
               "controlling for the market (SPY) over the same window.")
    real_docs = [d for d in docs if d.path.parent.name != "sample"]
    if not real_docs:
        st.info("No real filings loaded yet. Once `scripts/fetch_filings.py` has run, "
                "this tab scores sentiment per filing, fetches forward returns, and "
                "reports OLS coefficients, t-statistics and R² per window.")
    elif st.button("Run signal → return study", type="primary"):
        from finsight import returns as ret_mod
        rows = []
        prog = st.progress(0.0, "Scoring sentiment per filing…")
        for i, d in enumerate(real_docs):
            s = sentiment.score_text(d.full_text[:20000])
            rows.append({"ticker": d.ticker, "date": d.date, "signal": s.score})
            prog.progress((i + 1) / len(real_docs))
        prog.empty()
        with st.spinner("Fetching prices and running regressions…"):
            results = ret_mod.run_study(rows)
            for r in results:

                st.subheader(f"{r.window}-Day Forward Return")

                c1, c2, c3 = st.columns(3)

                c1.metric("R²", f"{r.r2:.3f}")
                c2.metric("RMSE", f"{r.rmse:.4f}")
                c3.metric("MAE", f"{r.mae:.4f}")

                c4, c5, c6 = st.columns(3)

                c4.metric("Coefficient (β)", f"{r.coef:.4f}")
                c5.metric("t-statistic", f"{r.t_stat:.2f}")
                c6.metric("Observations", r.n)

                if "mkt" in r.controls:
                    beta, t = r.controls["mkt"]
                    st.info(f"Market β = {beta:.3f} (t = {t:.2f})")

                st.divider()
        st.caption("Coefficient b is the marginal forward return per unit of sentiment, "
                   "holding the market return constant. |t| > 2 ≈ significant at 5%.")
