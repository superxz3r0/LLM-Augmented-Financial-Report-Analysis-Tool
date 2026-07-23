#prevents all type annotations from being evaluated immediately at definition time.
from __future__ import annotations 

from dataclasses import dataclass, field

import numpy as np

MARKET_PROXY = "SPY"


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
    controls: dict = field(default_factory=dict)#create a new dict object every time

    def summary(self) -> str:
        base = (
            f"{self.window}-day forward return ~ signal | "
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


def fetch_forward_returns(ticker: str, dates: list[str], windows=(5, 20)) -> dict:
    import pandas as pd
    import yfinance as yf

    #calculate the adjusted closing price
    px = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
    if px.empty:
        return {d: {} for d in dates}
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)

    out: dict[str, dict[int, float]] = {}
    for d in dates:
        ts = pd.Timestamp(d)

        #we need to let side = "right" cause we can only buy it after the signal 
        #day, and assume buy and sell both with closing price
        idx = px.index.searchsorted(ts, side="right")
        out[d] = {}
        
        #calculate the return rate for 5 and 20 days
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
        return RegressionResult(window, n, np.nan, np.nan, np.nan, np.nan)

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
        return RegressionResult(window, n, np.nan, np.nan, np.nan, np.nan)
    se = np.sqrt(np.diag(cov))
    t = np.where(se > 0, beta / se, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    ctrl_out = {k: (float(beta[2 + i]), float(t[2 + i])) for i, k in enumerate(ctrl)}
    return RegressionResult(window, n, float(beta[1]), float(beta[0]),
                            float(t[1]), float(r2), rmse, mae, ctrl_out)


def run_study(rows: list[dict], windows=(5, 20), market_control: bool = True
              ) -> list[RegressionResult]:
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

    results = []
    for w in windows:
        sig = np.array([r["signal"] for r in enriched])
        ret = np.array([r["returns"].get(w, np.nan) for r in enriched])
        controls = None
        if market_control:
            controls = {"mkt": np.array([r["market"].get(w, np.nan) for r in enriched])}
        results.append(ols(sig, ret, w, controls))
    return results
