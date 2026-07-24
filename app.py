from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="FinSight — Filing Analysis", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

from finsight import diff as diff_mod
from finsight import artifacts, rag, sentiment, s3links
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

    st.divider()
    st.subheader("Precomputed artifacts")
    _art = artifacts.summary()
    st.caption(f"sentiment: {_art['sentiment']} filings · "
               f"prices: {_art['returns_tickers']} tickers · "
               f"diffs: {_art['diff_pairs']} filing pairs")
    if _art["returns_generated"]:
        st.caption(f"prices generated {_art['returns_generated']}")
    if not (_art["sentiment"] and _art["returns_tickers"]):
        st.info("Artifacts missing — tabs will compute on the fly (slow). "
                "Build them with `python scripts/precompute.py --all` "
                "and rsync `data/derived/` to this machine.")

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
            items = artifacts.get_diff(artifacts.load_diffs(), t,
                                       str(old_doc.date), str(new_doc.date))
            if items is None:
                with st.spinner("Comparing filings (pair not precomputed)…"):
                    items = diff_mod.diff_documents(old_doc, new_doc)
            else:
                st.caption("precomputed diff")
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
    real_docs = [d for d in docs if d.path.parent.name != "sample"]
    st.caption(
        "Evaluate whether filing sentiment predicts future stock returns."
    )

    study_col1, study_col2 = st.columns([2,2])

    study_type = study_col1.radio(
        "Study Type",
        ["Overall Corpus", "Individual Company"],
        horizontal=True
    )

    selected_company = None

    if study_type == "Individual Company":
        selected_company = study_col2.selectbox(
            "Company",
            sorted({d.ticker for d in real_docs})
        )
    if not real_docs:
        st.info("No real filings loaded yet. Once `scripts/fetch_filings.py` has run, "
                "this tab scores sentiment per filing, fetches forward returns, and "
                "reports OLS coefficients, t-statistics and R² per window.")
    elif st.button("Run signal → return study", type="primary"):
        from finsight import returns as ret_mod

        docs_to_use = real_docs

        if selected_company is not None:
            docs_to_use = [
                d for d in real_docs
                if d.ticker == selected_company
            ]

        # ---- sentiment: precomputed artifact first, CPU fallback ----------
        art = artifacts.load_sentiment()
        keyed = [(d, t, artifacts.text_key(t))
                 for d, t in ((d, d.full_text[:artifacts.SENT_CHARS])
                              for d in docs_to_use)]
        missing = [(d, t, k) for d, t, k in keyed if k not in art]

        if missing:
            st.warning(
                f"{len(missing)} filings have no precomputed sentiment score — "
                "scoring FULL filing text on the local CPU now (roughly one to "
                "two minutes per filing). For the fast path, run "
                "`python scripts/precompute.py --sentiment --gpu` and sync "
                "`data/derived/` to this machine."
            )
            prog = st.progress(0.0, f"Scoring {len(missing)} filings on CPU…")
            for i, (d, t, k) in enumerate(missing):
                r = sentiment.score_text(
                    t, max_sentences=artifacts.SENT_MAX_SENTENCES)
                art[k] = {"score": r.score, "n_sentences": r.n_sentences,
                          "backend": r.backend, "ticker": d.ticker,
                          "date": str(d.date), "form": d.form}
                prog.progress((i + 1) / len(missing))
            prog.empty()
            artifacts.save_sentiment(art)

        rows = [{"ticker": d.ticker, "date": d.date,
                 "signal": art[k]["score"]} for d, _t, k in keyed]
        if len(rows) < 5:
            st.warning(
                f"Only {len(rows)} filings are available for this company. "
                "Regression results should be interpreted with caution."
            )
        with st.spinner("Fetching prices and running regressions…"):
            if selected_company:
                st.markdown(f"{selected_company} Signal → Return Study Analysing {len(rows)} filings")
            else:
                st.markdown("Overall Corpus Signal → Return Study")

            st.divider()

            ret_art = artifacts.load_returns()
            if ret_art and artifacts.returns_cover(ret_art, rows):
                results = artifacts.run_study_offline(rows, ret_art)
                st.caption("forward returns: precomputed prices "
                           f"({ret_art.get('generated_at', 'date unknown')})")
            else:
                if ret_art:
                    st.caption("precomputed prices don't cover every filing — "
                               "fetching live from yfinance instead")
                results = ret_mod.run_study(rows)
            for r in results:

                st.subheader(f"{r.window}-Day Forward Return")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "R²",
                    f"{r.r2:.3f}",
                    help="""
                Coefficient of Determination : Measures how much of the variation in future stock returns is explained by the model. The higher the better.

                Benchmark:
                
                • >0.60  Strong
                
                • 0.30–0.60  Moderate
                
                • <0.30  Weak
                """
                )

                c2.metric(
                    "RMSE",
                    f"{r.rmse:.4f}",
                    help="""
                Root Mean Squared Error : Measures the typical prediction error. Lower values indicate better predictive accuracy.
                """
                )

                c3.metric(
                    "MAE",
                    f"{r.mae:.4f}",
                    help="""
                Mean Absolute Error : Measures the average absolute difference between predicted and actual returns. Lower values indicate more accurate predictions.
                """
                )

                c4, c5, c6 = st.columns(3)

                c4.metric(
                    "Coefficient (β)",
                    f"{r.coef:.4f}",
                    help="""
                Regression coefficient : Represents the expected change in future return for a one-unit increase in the filing sentiment signal.

                • Positive value - Positive relationship
                
                • Negative value - Negative relationship
                """
                )

                c5.metric(
                    "t-statistic",
                    f"{r.t_stat:.2f}",
                    help="""
                It measures whether the relationship is statistically significant.

                Benchmark:
                
                • |t| > 2  → Significant
                
                • |t| < 2  → Weak statistical evidence
                """
                )

                c6.metric(
                    "Observations",
                    str(r.n),
                    help="""
                Number of company filings used to train the regression model. More observations generally improve the reliability of the estimated relationship.
                """
                )

                if abs(r.t_stat) >= 2:

                    if r.coef > 0:
                        verdict = "Positive"

                    else:
                        verdict = "Negative"

                else:
                    verdict = "Inconclusive"

                if "mkt" in r.controls:

                    beta, t = r.controls["mkt"]

                    c7, c8, c9 = st.columns(3)

                    c7.metric(
                        "Market β",
                        f"{beta:.3f}",
                        help="""
                Regression coefficient for the market return (SPY). Controls for overall market movement so that the filing signal is evaluated independently.
                """
                    )

                    c8.metric(
                        "Market t-stat",
                        f"{t:.2f}",
                        help="""
                Statistical significance of the market coefficient. Higher absolute values indicate stronger evidence that market movements influence future returns.
                """
                    )
                    
                c9.metric(
                    "Signal Assessment",
                    verdict,
                    help="""
                Overall assessment of the predictive power of the filing signal.

                • Positive - Positive relationship with statistically meaningful evidence.
                
                • Inconclusive - Relationship exists but evidence is not yet strong enough to draw firm conclusions.
                
                • Negative - Negative relationship supported by the regression.
                """
                )

                st.divider()
        st.caption("Coefficient b is the marginal forward return per unit of sentiment, "
                   "holding the market return constant. |t| > 2 ≈ significant at 5%.")
