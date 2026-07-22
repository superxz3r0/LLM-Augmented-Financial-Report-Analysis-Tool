from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path


import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="FinSight — Filing Analysis", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

from finsight import diff as diff_mod
from finsight import rag, sentiment, s3links  
from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.chunker import chunk_corpus
from finsight.index import build_index
from finsight.ingest import load_corpus

# Streamlit reruns the script in the same interpreter.  When source files are
# updated while file watching is disabled, an already-imported pre-upgrade
# module can otherwise survive and miss the disclosure regression API.
if not hasattr(diff_mod, "compute_disclosure_signals"):
    diff_mod = importlib.reload(diff_mod)

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


def document_text_prefix(doc, limit: int) -> str:
    """Build the same prefix as ``doc.full_text[:limit]`` without joining
    the remainder of a potentially very large filing first."""
    if limit <= 0:
        return ""
    parts = []
    remaining = limit
    for i, section in enumerate(doc.sections):
        piece = ("\n\n" if i else "") + section.text
        parts.append(piece[:remaining])
        remaining -= min(len(piece), remaining)
        if remaining == 0:
            break
    return "".join(parts)


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
            "6. **Correlate** — controlled event regressions with clustered inference")

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
            urls = {}
            for i, (c, _s) in enumerate(ans.sources, 1):
                m = re.match(r"\[.*\]\((.*)\)", s3links.citation_link(c))
                if m:
                    urls[i] = m.group(1)

            def linkify(text: str) -> str:
                return re.sub(
                    r"\[(\d+)\]",
                    lambda m: f"[[{m.group(1)}]]({urls[int(m.group(1))]})"
                              if int(m.group(1)) in urls else m.group(0),
                    text)

            st.markdown(linkify(esc(ans.text)))
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
                    st.markdown(f"**[{i}]** {s3links.citation_link(c)} · relevance {score:.2f}")
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
    a.metric(
        "Sentiment (MD&A)",
        f"{sig.sentiment_score:+.3f}",
        help=f"""
    Measures the overall tone of the Management Discussion & Analysis (MD&A) section using FinBERT.

    Range:
    
    • Positive (>0) - Optimistic management tone
    
    • Negative (<0) - Pessimistic management tone
    
    • Around 0 - Neutral

    """
    )

    b.metric(
        "Risk Factor Signals",
        sig.risk_factor_count,
        help="""
    Counts the number of risk-related disclosures identified in the filing. Higher values generally indicate that management discusses more risks or uncertainties. This is a rule-based extraction and should be interpreted as an indicator rather than a direct measure of company risk.
    """
    )

    c.metric(
        "Revenue Guidance Hits",
        len(sig.revenue_guidance),
        help="""
    Number of revenue guidance statements detected in the filing. More guidance statements generally indicate greater discussion of future revenue performance.
    
    Examples include: Revenue forecasts, Sales outlook, Growth expectations.
    """
    )

    d4.metric(
        "Capex Guidance Hits",
        len(sig.capex_guidance),
        help="""
    Number of capital expenditure (CapEx) guidance statements detected. CapEx guidance helps assess future investment strategy.

    Examples include: Planned investments, Expansion spending, Infrastructure or equipment expenditure
    """
    )
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

    st.caption("Computed financial analysis: Health Checks, Ratios and YoY growth")
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
        
        DISPLAY = {
            "gross_margin": "Gross Margin",
            "operating_margin": "Operating Margin",
            "net_margin": "Net Margin",
            "current_ratio": "Current Ratio",
            "debt_to_equity": "Debt-to-Equity",
            "roe": "Return on Equity (ROE)",
            "roa": "Return on Assets (ROA)",
            "asset_turnover": "Asset Turnover",
            "fcf": "Free Cash Flow",
            "fcf_margin": "Free Cash Flow Margin",
            "revenue": "Revenue",
            "operating_income": "Operating Income",
            "net_income": "Net Income",
            "operating_cash_flow": "Operating Cash Flow"
        }
        
        st.markdown("### Health Check")
        st.markdown("**Overall Health Check**")

        flags = metrics_mod.health_check(periods)

        # Count severities
        red = sum(f.severity == "red" for f in flags)
        amber = sum(f.severity == "amber" for f in flags)
        green = sum(f.severity == "green" for f in flags)

        # Overall verdict
        if red >= 2:
            verdict = "🔴 Weak"
            verdict_msg = "Multiple financial metrics require attention."
        elif red == 1:
            verdict = "🟡 Moderate"
            verdict_msg = "Overall financial health is acceptable, but some metrics should be monitored."
        else:
            verdict = "🟢 Strong"
            verdict_msg = "Financial indicators appear healthy overall."

        # Overall summary card
        with st.container(border=True):
            st.markdown(f"### {verdict}")
            st.write(verdict_msg)

        st.markdown("**Individual Checks**")

        icon = {
            "red": "🔴",
            "amber": "🟡",
            "green": "🟢"
        }

        status = {
            "red": "Critical",
            "amber": "Warning",
            "green": "Healthy"
        }

        with st.container(border=True):

            cols = st.columns(len(flags))

            for col, f in zip(cols, flags):

                metric_name = DISPLAY.get(f.metric, f.metric.replace("_", " ").title())

                col.markdown(f"## {icon[f.severity]}")
                col.markdown(f"**{metric_name}**")
                col.caption(status[f.severity])
                col.write(f.message)

        st.divider()
            
        metric_help = {
            "gross_margin":
                "Percentage of revenue remaining after direct production costs. "
                "Higher values indicate stronger profitability and pricing power.",

            "operating_margin":
                "Operating income as a percentage of revenue before interest and taxes. "
                "Measures how efficiently the company's core business generates profit.",

            "net_margin":
                "Percentage of revenue retained as net profit after all expenses. "
                "Higher margins indicate better overall financial performance.",

            "current_ratio":
                "Current Assets ÷ Current Liabilities. "
                "A value above 1 generally indicates the company can meet its short-term obligations.",

            "debt_to_equity":
                "Compares total debt to shareholders' equity. "
                "Lower values generally indicate lower financial risk and reliance on borrowing.",

            "roe":
                "Return on Equity measures how effectively shareholder investment generates profit. "
                "Higher ROE generally indicates stronger profitability and management efficiency.",

            "roa":
                "Return on Assets measures how efficiently company assets generate earnings. "
                "Higher values indicate better utilization of available resources.",

            "asset_turnover":
                "Measures how efficiently assets are used to generate revenue. "
                "Higher values indicate more productive use of company assets.",

            "fcf":
                "Free Cash Flow is the cash remaining after operating and capital expenses. "
                "Positive FCF indicates the company has cash available for growth, debt repayment, or dividends.",

            "fcf_margin":
                "Free Cash Flow expressed as a percentage of revenue. "
                "Higher values indicate stronger cash generation from sales.",

            "revenue":
                "Total income generated from the company's core business operations. "
                "Consistent revenue growth generally reflects increasing business demand.",

            "operating_income":
                "Profit earned from normal business operations before interest and taxes. "
                "Shows how profitable the company's core operations are.",

            "net_income":
                "The company's final profit after all expenses, taxes, and interest. "
                "Often referred to as the 'bottom line' of the income statement.",

            "operating_cash_flow":
                "Cash generated from the company's day-to-day business operations. "
                "Positive operating cash flow indicates the business is generating sustainable cash."
        }

        st.markdown("### Metric Guide")

        st.caption("Learn about a financial metric :")

        selected_metric = st.selectbox(
            "Metric",
            list(metric_help.keys()),
            format_func=lambda x: DISPLAY.get(x, x),
            label_visibility="collapsed"
        )

        with st.container(border=True):
            st.markdown(f"**{DISPLAY[selected_metric]}**")
            st.write(metric_help[selected_metric])

        ratio_rows = {p.period: metrics_mod.compute_ratios(p) for p in periods}
        df = pd.DataFrame(ratio_rows)
        df.rename(index=DISPLAY, inplace=True)
        pct = ["gross_margin", "operating_margin", "net_margin", "roe", "roa", "fcf_margin"]

        def _fmt(idx, v):
            if v is None or (isinstance(v, float) and v != v):
                return "—"
            if idx in pct:
                return f"{v:.1%}"
            return f"{v:,.0f}" if idx == "fcf" else f"{v:.2f}"
        st.markdown("**Ratios**")
        df.index.name = "Metric"
        styled = pd.DataFrame(
            {col: [_fmt(idx, df.at[idx, col]) for idx in df.index] for col in df.columns},
            index=df.index,
            dtype="object",
        )
        trend = []

        for metric in df.index:

            vals = df.loc[metric].dropna().tolist()

            if len(vals) < 2:
                trend.append("—")

            elif vals[-1] > vals[0]:
                trend.append("📈 Improving")

            elif vals[-1] < vals[0]:
                trend.append("📉 Declining")

            else:
                trend.append("➜ Stable")

        styled["Trend"] = trend
        
        st.dataframe(styled, use_container_width=True)
        

        g = metrics_mod.growth_analysis(periods)
        if any(any(v is not None for v in vals) for vals in g.values()):
            st.markdown("**YoY growth**")
            glabels = [f"{a.period} - {b.period}" for a, b in zip(periods, periods[1:])]
            gdf = pd.DataFrame(g, index=glabels).T
            gdf.index.name = "Metric"
            gdf.rename(index=DISPLAY, inplace=True)

            trend = []

            for metric in gdf.index:

                vals = pd.to_numeric(gdf.loc[metric], errors="coerce").dropna().tolist()

                if len(vals) < 2:
                    trend.append("➜ Stable")

                else:
                    first = vals[0]
                    last = vals[-1]

                    # Small tolerance to avoid tiny fluctuations
                    eps = 0.01

                    if last > first + eps:
                        trend.append("📈 Accelerating")

                    elif last < first - eps:
                        trend.append("📉 Decelerating")

                    else:
                        trend.append("➜ Stable")

            gdf["Trend"] = trend
            
            st.dataframe(
                gdf.map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}" if isinstance(v, (int, float)) else v),
                use_container_width=True,
            )

            st.divider()

            st.subheader("Financial Trend")

            graph_type = st.radio(
                "Display",
                ["Financial Ratios", "YoY Growth"],
                horizontal=True
            )

            if graph_type == "Financial Ratios":

                metric = st.selectbox(
                    "Choose Ratio",
                    df.index.tolist(),
                    key="ratio_plot"
                )

                plot_df = pd.DataFrame(
                    {
                        metric: df.loc[metric]
                    }
                )

            else:

                metric = st.selectbox(
                    "Choose Growth Metric",
                    gdf.index.tolist(),
                    key="growth_plot"
                )

                growth_series = (
                    pd.to_numeric(
                        gdf.loc[metric].drop("Trend"),
                        errors="coerce"
                    ) * 100
                )

                plot_df = pd.DataFrame(
                    {
                        metric: growth_series
                    }
                )

            st.line_chart(plot_df)

        
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
    import pandas as pd

    from finsight import returns as ret_mod
    from finsight import store as store_mod

    if not hasattr(ret_mod, "MarketDataUnavailable"):
        ret_mod = importlib.reload(ret_mod)

    real_docs = [d for d in docs if d.path.parent.name != "sample"]
    st.caption(
        "Event-level regressions test whether FinBERT sentiment and normalized "
        "disclosure changes predict 5- and 20-trading-day forward returns."
    )

    study_col1, study_col2, study_col3 = st.columns([2, 2, 2])
    study_type = study_col1.radio(
        "Study type",
        ["Overall Corpus", "Individual Company"],
        horizontal=True,
    )

    selected_company = None
    if study_type == "Individual Company":
        company_options = sorted({d.ticker for d in real_docs})
        selected_company = study_col2.selectbox(
            "Company",
            company_options,
            disabled=not company_options,
        )
        study_col3.text_input(
            "Robust covariance",
            value="Clustered by event date",
            disabled=True,
            help=(
                "Ticker clustering is not identifiable for one company. "
                "Event-date clustering is HC1-equivalent when dates are unique."
            ),
        )
        cluster_by = "date"
    else:
        cluster_choice = study_col3.selectbox(
            "Robust covariance",
            ["Clustered by ticker", "Clustered by event date"],
            help=(
                "Ticker clustering allows arbitrary correlation within a company. "
                "Event-date clustering allows common shocks across companies."
            ),
        )
        cluster_by = "ticker" if cluster_choice == "Clustered by ticker" else "date"

    include_joint = st.checkbox(
        "Include joint sentiment + disclosure-change model",
        value=True,
        help="Runs the two required single-signal models plus a joint robustness model.",
    )
    requested_models = (
        ("sentiment", "disclosure", "joint")
        if include_joint
        else ("sentiment", "disclosure")
    )

    docs_to_use = real_docs
    if selected_company is not None:
        docs_to_use = [d for d in real_docs if d.ticker == selected_company]

    corpus_digest = hashlib.sha256()
    for document in docs_to_use:
        try:
            stat = document.path.stat()
            file_revision = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            file_revision = "unavailable"
        corpus_digest.update(
            f"{document.doc_id}|{document.date}|{document.form}|{file_revision}\n".encode(
                "utf-8"
            )
        )

    study_key = (
        study_type,
        selected_company,
        cluster_by,
        include_joint,
        corpus_digest.hexdigest(),
        store_mod.REGRESSION_SENTIMENT_VERSION,
        store_mod.regression_sentiment_model_key(),
        getattr(diff_mod, "DISCLOSURE_SIGNAL_VERSION", 1),
        getattr(ret_mod, "_FORWARD_RETURN_CACHE_VERSION", 1),
    )
    state_name = "_signal_return_regression_study"

    if not real_docs:
        st.info(
            "No real filings are loaded. Run scripts/fetch_filings.py before "
            "starting the signal-to-return study."
        )
    elif st.button("Run controlled regression study", type="primary"):
        # Never leave a stale successful study visible beneath a failed rerun.
        st.session_state.pop(state_name, None)
        prog = st.progress(0.0, "Preparing documents and checking sentiment cache...")
        try:
            prepared = []
            for index, document in enumerate(docs_to_use):
                text = document_text_prefix(
                    document, store_mod.REGRESSION_TEXT_CHARS
                )
                prepared.append((document, text))
                prog.progress(
                    0.08 * (index + 1) / len(docs_to_use),
                    f"Preparing documents... {index + 1}/{len(docs_to_use)}",
                )

            def sentiment_progress(done: int, total: int) -> None:
                fraction = done / total if total else 1.0
                prog.progress(
                    0.10 + 0.64 * fraction,
                    f"FinBERT sentence batches... {done:,}/{total:,}",
                )

            scores, cached_count, computed_count = (
                store_mod.score_regression_sentiments_cached(
                    [(document.doc_id, text) for document, text in prepared],
                    progress=sentiment_progress,
                )
            )

            prog.progress(
                0.78,
                "Computing normalized disclosure changes against prior same-form filings...",
            )
            disclosure_results = diff_mod.compute_disclosure_signals(docs_to_use)
            disclosure_by_doc = {
                result.doc_id: result.disclosure_change
                for result in disclosure_results
            }

            rows = [
                {
                    "doc_id": document.doc_id,
                    "ticker": document.ticker,
                    "date": document.date,
                    "form": document.form,
                    "sector": ret_mod.sector_for_ticker(document.ticker),
                    "sentiment": score.score,
                    "disclosure_change": disclosure_by_doc.get(document.doc_id),
                }
                for (document, _text), score in zip(prepared, scores)
            ]
            events = ret_mod.aggregate_event_rows(rows)
            merged_documents = len(rows) - len(events)
            disclosure_events = sum(
                math.isfinite(float(event["disclosure_change"]))
                for event in events
                if event.get("disclosure_change") is not None
            )
            sector_events = sum(
                event.get("sector") not in (None, "", "Unknown")
                for event in events
            )

            prog.progress(
                0.92, "Fetching prices and fitting robust controlled models..."
            )
            with st.spinner(
                "Fetching prices once per ticker and fitting clustered regressions..."
            ):
                results = ret_mod.run_study(
                    rows,
                    models=requested_models,
                    cluster_by=cluster_by,
                    market_control=True,
                    sector_control=True,
                    form_control=True,
                )
            if not results or max((result.n for result in results), default=0) == 0:
                raise ret_mod.MarketDataUnavailable(
                    "No event has both a stock forward return and the SPY "
                    "market-control return. The regression was not saved."
                )
        except ret_mod.MarketDataUnavailable as exc:
            st.error(f"Market data unavailable: {exc}")
            st.info(
                "The FinBERT and disclosure caches are intact. Allow outbound "
                "access to Yahoo Finance once to populate the persistent "
                "forward-return cache in data/finsight.db; later app and cloud "
                "runs can reuse it without another price request."
            )
        except Exception as exc:
            st.error(
                "Regression study preparation or fitting failed: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            prog.progress(1.0, "Regression study complete.")
            priced_events_by_window = {
                result.window: result.n
                for result in results
                if result.model == "sentiment"
            }
            st.session_state[state_name] = {
                "key": study_key,
                "results": results,
                "raw_documents": len(rows),
                "unique_events": len(events),
                "merged_documents": merged_documents,
                "disclosure_documents": sum(
                    result.available for result in disclosure_results
                ),
                "disclosure_events": disclosure_events,
                "sector_events": sector_events,
                "cached_sentiment": cached_count,
                "computed_sentiment": computed_count,
                "company": selected_company,
                "cluster_requested": cluster_by,
                "priced_events_by_window": priced_events_by_window,
            }
        finally:
            prog.empty()

    study = st.session_state.get(state_name)
    if study is not None and study.get("key") == study_key:
        results = study["results"]

        if selected_company is not None:
            st.markdown(
                f"### {selected_company} controlled signal-to-return study"
            )
            st.info(
                "Sector is constant within one company and is therefore absorbed "
                "by the intercept. Robust inference is clustered by event date."
            )
        else:
            st.markdown("### Overall corpus controlled signal-to-return study")

        summary_columns = st.columns(6)
        summary_columns[0].metric("Raw documents", study["raw_documents"])
        summary_columns[1].metric("Unique events", study["unique_events"])
        summary_columns[2].metric("Merged duplicates", study["merged_documents"])
        priced_by_window = study["priced_events_by_window"]
        priced_counts = sorted(priced_by_window.values())
        priced_display = (
            str(priced_counts[0])
            if priced_counts[0] == priced_counts[-1]
            else f"{priced_counts[0]}–{priced_counts[-1]}"
        )
        summary_columns[3].metric("Priced events", priced_display)
        summary_columns[4].metric(
            "Disclosure events", study["disclosure_events"]
        )
        summary_columns[5].metric(
            "Sector coverage",
            f'{study["sector_events"]}/{study["unique_events"]}',
        )
        if any(
            count < study["unique_events"]
            for count in priced_by_window.values()
        ):
            coverage = ", ".join(
                f"{window}-day {count}/{study['unique_events']}"
                for window, count in sorted(priced_by_window.items())
            )
            st.warning(
                "Complete stock/SPY price-pair coverage: " + coverage + ". "
                "Incomplete events are excluded listwise."
            )
        st.caption(
            f'Sentiment cache: {study["cached_sentiment"]} reused, '
            f'{study["computed_sentiment"]} computed. '
            f'Disclosure comparisons: {study["disclosure_documents"]} documents.'
        )

        sector_formula = (
            "sector fixed effects"
            if selected_company is None
            else "sector absorbed by the intercept"
        )
        model_metadata = {
            "sentiment": {
                "label": "Sentiment",
                "formula": (
                    f"return ~ sentiment + market + {sector_formula} "
                    "+ document-form fixed effects"
                ),
                "focal": ("sentiment",),
            },
            "disclosure": {
                "label": "Disclosure change",
                "formula": (
                    f"return ~ disclosure_change + market + {sector_formula}"
                ),
                "focal": ("disclosure_change",),
            },
            "joint": {
                "label": "Joint",
                "formula": (
                    "return ~ sentiment + disclosure_change + market "
                    f"+ {sector_formula} + document-form fixed effects"
                ),
                "focal": ("sentiment", "disclosure_change"),
            },
        }

        def finite_number(value) -> bool:
            try:
                return math.isfinite(float(value))
            except (TypeError, ValueError):
                return False

        def display_number(value, digits: int = 4) -> str:
            return f"{float(value):.{digits}f}" if finite_number(value) else "—"

        def display_p_value(value) -> str:
            if not finite_number(value):
                return "—"
            number = float(value)
            return "<0.0001" if number < 0.0001 else f"{number:.4f}"

        def assessment(estimate) -> str:
            if estimate is None or not finite_number(estimate.p_value):
                return "Not estimable"
            if (
                estimate.p_value < 0.05
                and estimate.ci_low > 0
                and estimate.ci_high > 0
            ):
                return "Positive — statistically significant"
            if (
                estimate.p_value < 0.05
                and estimate.ci_low < 0
                and estimate.ci_high < 0
            ):
                return "Negative — statistically significant"
            return "Not statistically significant"

        comparison_rows = []
        for result in results:
            meta = model_metadata[result.model]
            for term in meta["focal"]:
                estimate = result.coefficients.get(term)
                comparison_rows.append(
                    {
                        "Window": f"{result.window}-day",
                        "Model": meta["label"],
                        "Signal": term,
                        "Coefficient": (
                            display_number(estimate.coefficient)
                            if estimate is not None else "—"
                        ),
                        "Robust SE": (
                            display_number(estimate.std_error)
                            if estimate is not None else "—"
                        ),
                        "t-stat": (
                            display_number(estimate.t_stat, 3)
                            if estimate is not None else "—"
                        ),
                        "p-value": (
                            display_p_value(estimate.p_value)
                            if estimate is not None else "—"
                        ),
                        "95% CI": (
                            f"[{display_number(estimate.ci_low)}, "
                            f"{display_number(estimate.ci_high)}]"
                            if estimate is not None else "—"
                        ),
                        "N events": result.n,
                        "Adjusted R²": display_number(result.adjusted_r2, 3),
                        "Assessment": assessment(estimate),
                    }
                )

        st.markdown("#### Model comparison")
        comparison_frame = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_frame, hide_index=True, use_container_width=True)
        st.download_button(
            "Download model comparison CSV",
            comparison_frame.to_csv(index=False).encode("utf-8"),
            file_name="finsight_regression_model_comparison.csv",
            mime="text/csv",
        )

        windows = sorted({result.window for result in results})
        window_tabs = st.tabs([f"{window}-day return" for window in windows])
        for window_tab, window in zip(window_tabs, windows):
            with window_tab:
                window_results = [
                    result for result in results if result.window == window
                ]
                for result in window_results:
                    meta = model_metadata[result.model]
                    with st.container(border=True):
                        st.markdown(f"#### {meta['label']} model")
                        st.code(meta["formula"], language=None)

                        metrics = st.columns(6)
                        metrics[0].metric("N events", result.n)
                        metrics[1].metric(
                            "Adjusted R²", display_number(result.adjusted_r2, 3)
                        )
                        metrics[2].metric("R²", display_number(result.r2, 3))
                        metrics[3].metric("RMSE", display_number(result.rmse))
                        metrics[4].metric("MAE", display_number(result.mae))
                        metrics[5].metric(
                            "Clusters",
                            result.n_clusters if result.n_clusters is not None else "—",
                        )

                        for term in meta["focal"]:
                            estimate = result.coefficients.get(term)
                            verdict = assessment(estimate)
                            if verdict.startswith("Positive"):
                                st.success(f"{term}: {verdict}")
                            elif verdict.startswith("Negative"):
                                st.warning(f"{term}: {verdict}")
                            elif verdict == "Not estimable":
                                st.warning(f"{term}: {verdict}")
                            else:
                                st.info(f"{term}: {verdict}")

                        coefficient_rows = []
                        for term, estimate in result.coefficients.items():
                            coefficient_rows.append(
                                {
                                    "Term": term,
                                    "Coefficient": (
                                        estimate.coefficient
                                        if finite_number(estimate.coefficient)
                                        else None
                                    ),
                                    "Robust SE": (
                                        estimate.std_error
                                        if finite_number(estimate.std_error)
                                        else None
                                    ),
                                    "t-statistic": (
                                        estimate.t_stat
                                        if finite_number(estimate.t_stat)
                                        else None
                                    ),
                                    "p-value": (
                                        estimate.p_value
                                        if finite_number(estimate.p_value)
                                        else None
                                    ),
                                    "CI lower": (
                                        estimate.ci_low
                                        if finite_number(estimate.ci_low)
                                        else None
                                    ),
                                    "CI upper": (
                                        estimate.ci_high
                                        if finite_number(estimate.ci_high)
                                        else None
                                    ),
                                }
                            )
                        coefficient_frame = pd.DataFrame(coefficient_rows)
                        if coefficient_rows:
                            st.dataframe(
                                coefficient_frame,
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    "Coefficient": st.column_config.NumberColumn(
                                        format="%.6f"
                                    ),
                                    "Robust SE": st.column_config.NumberColumn(
                                        format="%.6f"
                                    ),
                                    "t-statistic": st.column_config.NumberColumn(
                                        format="%.3f"
                                    ),
                                    "p-value": st.column_config.NumberColumn(
                                        format="%.4f"
                                    ),
                                    "CI lower": st.column_config.NumberColumn(
                                        format="%.6f"
                                    ),
                                    "CI upper": st.column_config.NumberColumn(
                                        format="%.6f"
                                    ),
                                },
                            )
                            st.download_button(
                                f"Download {window}-day {result.model} coefficients",
                                coefficient_frame.to_csv(index=False).encode("utf-8"),
                                file_name=(
                                    f"finsight_{window}d_{result.model}_coefficients.csv"
                                ),
                                mime="text/csv",
                                key=f"download_{window}_{result.model}_{study_key}",
                            )
                        else:
                            st.warning(
                                "This specification could not be estimated with "
                                "the available complete events."
                            )

                        reference_text = ", ".join(
                            f"{name}: {value}"
                            for name, value in result.reference_categories.items()
                        )
                        cluster_text = (
                            f"{result.covariance}; clustered by "
                            f"{result.cluster_by} (G={result.n_clusters})"
                            if result.cluster_by is not None
                            else result.covariance
                        )
                        st.caption(
                            f"{cluster_text}. "
                            + (
                                f"Reference categories — {reference_text}."
                                if reference_text
                                else "No varying categorical control in this sample."
                            )
                        )
                        for warning in result.warnings:
                            st.warning(warning)

        with st.expander("Methodology and interpretation"):
            st.markdown(
                """
- **Event unit:** documents sharing the same ticker and filing date are
  collapsed to one event. Available sentiment and disclosure scores are
  averaged, and the realized return is counted once.
- **Disclosure-change signal:** each 10-K/10-Q is compared with the strictly
  previous filing of the same ticker and form. Item-aligned token-multiset
  Dice similarity is converted to a bounded 0–1 score; new/removed and
  substantive changes receive full weight, minor changes receive 0.25, and
  boilerplate-equivalent text receives zero weight.
- **Missing comparisons:** the first filing in each ticker/form sequence and
  transcripts have no disclosure score and are excluded only from models that
  require that signal.
- **Controls:** the corpus models include SPY market returns and k−1 sector
  dummy variables. Sentiment and joint models also include k−1 event-form
  dummies. Reference categories are reported with every model.
- **Forward return:** the first trading session strictly after the document
  date is the baseline; the endpoint is 5 or 20 trading sessions later.
- **Inference:** one-way CR1 cluster-robust standard errors use a finite-sample
  correction and Student-t p-values. Corpus studies default to ticker
  clusters; one-company studies use event-date clusters. With one observation
  per event date this is HC1-equivalent and does not model serial correlation
  across the company's filing events.
- **Assessment:** significance is based on the exact robust p-value and 95%
  confidence interval. “Not statistically significant” does not imply that a
  relationship exists or that the null hypothesis has been proved.
                """
            )
