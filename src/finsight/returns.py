from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

MARKET_PROXY = "SPY"

# Filing sentiment and transcript sentiment behave differently as return
# predictors: periodic filings (10-K/10-Q) are dense, backward-looking
# accounting disclosures, whereas earnings-call transcripts are
# forward-looking management commentary with a very different tone
# distribution and event-timing profile. Pooling them into one regression
# (the old behaviour) blends two distinct signals and washes the coefficient
# out toward zero — the same reason the RAG layer filters retrieval per
# company rather than searching one undifferentiated pool. We therefore run
# one regression *per document group*: periodic filings together, transcripts
# on their own.
DOC_GROUPS: dict[str, str] = {
    "10-K": "Periodic Filings (10-K / 10-Q)",
    "10-Q": "Periodic Filings (10-K / 10-Q)",
    "TRANSCRIPT": "Earnings-Call Transcripts",
}
DEFAULT_GROUP = "Other Documents"


def doc_group(form: str) -> str:
    """Map a raw filing `form` string to its regression group label.

    10-K and 10-Q collapse into a single "Periodic Filings" group;
    transcripts form their own group; anything else falls back to a generic
    bucket so it is never silently dropped.
    """
    return DOC_GROUPS.get((form or "").upper(), DEFAULT_GROUP)


@dataclass
class RegressionResult:
    window: int
    n: int
    coef: float
    intercept: float
    t_stat: float
    r2: float
    rmse: float
    mae: float
    controls: dict = field(default_factory=dict)
    group: str = ""

    def summary(self) -> str:
        prefix = f"[{self.group}] " if self.group else ""
        base = (
            prefix
            + f"{self.window}-day forward return ~ signal | "
            f"n={self.n}, "
            f"b={self.coef:.4f} "
            f"(t={self.t_stat:.2f}), "
            f"R²={self.r2:.3f}, "
            f"RMSE={self.rmse:.4f}, "
            f"MAE={self.mae:.4f}"
        )
        for name, (c, t) in self.controls.items():
            base += f" | {name}: {c:.3f} (t={t:.2f})"
        return base


@functools.lru_cache(maxsize=128)
def _price_history(ticker: str):
    """Full daily close history for a ticker, cached per process.

    The signal->returns study calls fetch_forward_returns once per ticker per
    window set; without this cache, re-running the study (or an "overall
    corpus" study whose tickers overlap an earlier per-company one) re-downloads
    each ticker's entire price history from Yahoo every time.
    """
    import yfinance as yf

    px = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px


def fetch_forward_returns(ticker: str, dates: list[str], windows=(5, 20)) -> dict:
    import pandas as pd

    px = _price_history(ticker)
    if px.empty:
        return {d: {} for d in dates}

    out: dict[str, dict[int, float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        idx = px.index.searchsorted(ts, side="right")
        out[d] = {}
        for w in windows:
            if idx + w < len(px):
                p0, p1 = float(px.iloc[idx]), float(px.iloc[idx + w])
                out[d][w] = p1 / p0 - 1.0
    return out


def ols(signal: np.ndarray, ret: np.ndarray, window: int,
        controls: dict[str, np.ndarray] | None = None) -> RegressionResult:
    x = np.asarray(signal, float)
    y = np.asarray(ret, float)
    ctrl = {k: np.asarray(v, float) for k, v in (controls or {}).items()}

    mask = ~(np.isnan(x) | np.isnan(y))
    for v in ctrl.values():
        mask &= ~np.isnan(v)
    x, y = x[mask], y[mask]
    ctrl = {k: v[mask] for k, v in ctrl.items()}
    n = len(x)
    if n < 3 + len(ctrl):
        return RegressionResult(window, n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    cols = [np.ones(n), x] + [ctrl[k] for k in ctrl]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    pred = X @ beta

    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    mae = float(np.mean(np.abs(y - pred)))
    dof = n - X.shape[1]
    sigma2 = resid @ resid / max(dof, 1)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return RegressionResult(window, n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    se = np.sqrt(np.diag(cov))
    t = np.where(se > 0, beta / se, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    ctrl_out = {k: (float(beta[2 + i]), float(t[2 + i])) for i, k in enumerate(ctrl)}
    return RegressionResult(window, n, float(beta[1]), float(beta[0]),
                            float(t[1]), float(r2), rmse, mae, ctrl_out)


def _enrich(rows: list[dict], windows, market_control: bool) -> list[dict]:
    """Attach forward returns (and market returns) to each row.

    Prices are fetched per ticker (and the market proxy once), so enrichment
    is done over the full row set up front and shared across groups — this
    avoids re-downloading the same ticker once per document group.
    """
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)

    all_dates = sorted({r["date"] for r in rows})
    market = fetch_forward_returns(MARKET_PROXY, all_dates, windows) if market_control else {}

    enriched = []
    for ticker, items in by_ticker.items():
        fr = fetch_forward_returns(ticker, [r["date"] for r in items], windows)
        for r in items:
            r2 = dict(r)
            r2["returns"] = fr.get(r["date"], {})
            r2["market"] = market.get(r["date"], {})
            enriched.append(r2)
    return enriched


def _regress(enriched: list[dict], windows, market_control: bool, group: str = ""
             ) -> list[RegressionResult]:
    results = []
    for w in windows:
        sig = np.array([r["signal"] for r in enriched])
        ret = np.array([r["returns"].get(w, np.nan) for r in enriched])
        controls = None
        if market_control:
            controls = {"mkt": np.array([r["market"].get(w, np.nan) for r in enriched])}
        res = ols(sig, ret, w, controls)
        res.group = group
        results.append(res)
    return results


def run_study(rows: list[dict], windows=(5, 20), market_control: bool = True
              ) -> list[RegressionResult]:
    """Single pooled regression over all rows (legacy behaviour)."""
    enriched = _enrich(rows, windows, market_control)
    return _regress(enriched, windows, market_control)


def run_study_grouped(rows: list[dict], windows=(5, 20), market_control: bool = True
                      ) -> dict[str, list[RegressionResult]]:
    """Run one regression per document group.

    Each row must carry a "form" key ("10-K", "10-Q", "TRANSCRIPT", …); rows
    are partitioned by `doc_group(form)` so periodic filings and transcripts
    are modelled separately rather than pooled. Returns an ordered mapping of
    group label -> per-window results. Groups are emitted in a stable order
    (periodic filings first, then transcripts, then any fallback bucket).
    """
    enriched = _enrich(rows, windows, market_control)

    grouped: dict[str, list[dict]] = {}
    for r in enriched:
        grouped.setdefault(doc_group(r.get("form", "")), []).append(r)

    order = ["Periodic Filings (10-K / 10-Q)", "Earnings-Call Transcripts", DEFAULT_GROUP]
    ordered_labels = [g for g in order if g in grouped]
    ordered_labels += [g for g in grouped if g not in order]

    return {
        label: _regress(grouped[label], windows, market_control, group=label)
        for label in ordered_labels
    }
