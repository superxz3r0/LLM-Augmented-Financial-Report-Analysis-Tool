"""Forward-return retrieval and event-level regression inference.

The public :func:`run_study` entry point deliberately keeps the original
``signal``-only input working.  Richer callers can additionally provide
``disclosure_change``, ``sector`` and ``form`` for each document.  Documents
filed by the same company on the same date are collapsed into one event before
prices are joined, so a single realised return is never counted twice.

Regression standard errors in ``run_study`` are one-way cluster robust.  The
default clusters by ticker for a corpus study and falls back to event date when
only one ticker is present.  :func:`ols` remains available as the small,
classical-OLS compatibility helper used by earlier code and tests.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import DATA_DIR

MARKET_PROXY = "SPY"
_YFINANCE_CACHE_LOCK = Lock()
_YFINANCE_CACHE_CONFIGURED = False
_FORWARD_RETURN_CACHE_VERSION = 1
_PRICE_TIMEOUT_SECONDS = 12


class MarketDataUnavailable(RuntimeError):
    """Raised when neither live prices nor persistent returns are available."""

# Stable, deployment-independent GICS-style sectors for the real corpus.
# A caller-supplied ``sector`` always takes precedence, so this mapping is a
# deterministic fallback rather than a hidden external data dependency.
TICKER_SECTORS: dict[str, str] = {
    "AAPL": "Information Technology",
    "AMD": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "BAC": "Financials",
    "COST": "Consumer Staples",
    "CRM": "Information Technology",
    "CVX": "Energy",
    "DIS": "Communication Services",
    "GOOGL": "Communication Services",
    "GS": "Financials",
    "INTC": "Information Technology",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "KO": "Consumer Staples",
    "META": "Communication Services",
    "MSFT": "Information Technology",
    "NFLX": "Communication Services",
    "NVDA": "Information Technology",
    "PEP": "Consumer Staples",
    "PFE": "Health Care",
    "TSLA": "Consumer Discretionary",
    "UNH": "Health Care",
    "WMT": "Consumer Staples",
    "XOM": "Energy",
}


def sector_for_ticker(ticker: str, default: str = "Unknown") -> str:
    """Return the bundled sector classification without making a web call."""
    return TICKER_SECTORS.get(str(ticker).strip().upper(), default)


def _configure_yfinance_cache(yf) -> Path:
    """Point yfinance's SQLite caches at a writable project directory."""
    global _YFINANCE_CACHE_CONFIGURED

    cache_dir = Path(
        os.environ.get("FINSIGHT_YFINANCE_CACHE_DIR", DATA_DIR / "yfinance-cache")
    ).expanduser()
    if _YFINANCE_CACHE_CONFIGURED:
        return cache_dir

    with _YFINANCE_CACHE_LOCK:
        if not _YFINANCE_CACHE_CONFIGURED:
            cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_dir))
            _YFINANCE_CACHE_CONFIGURED = True
    return cache_dir


def _forward_return_cache_path() -> Path:
    """Keep market results in the same portable database as signal caches."""
    return DATA_DIR / "finsight.db"


def _ensure_forward_return_table(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS forward_return_cache (
            ticker TEXT NOT NULL,
            event_date TEXT NOT NULL,
            window INTEGER NOT NULL,
            version INTEGER NOT NULL,
            value REAL NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, event_date, window, version)
        )"""
    )


def _cached_forward_returns(
    ticker: str,
    dates: Sequence[str],
    windows: Sequence[int],
) -> dict[str, dict[int, float]]:
    requested_dates = set(dates)
    requested_windows = {int(window) for window in windows}
    output = {date: {} for date in dates}
    path = _forward_return_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as con:
            _ensure_forward_return_table(con)
            rows = con.execute(
                """SELECT event_date, window, value
                   FROM forward_return_cache
                   WHERE ticker=? AND version=?""",
                (ticker, _FORWARD_RETURN_CACHE_VERSION),
            ).fetchall()
    except (OSError, sqlite3.Error):
        # A read-only deployment can still use live data; lack of an optional
        # cache must not make the price provider fail.
        return output
    for event_date, window, value in rows:
        window = int(window)
        if event_date in requested_dates and window in requested_windows:
            output[event_date][window] = float(value)
    return output


def _persist_forward_returns(
    ticker: str,
    values: Mapping[str, Mapping[int, float]],
    source: str,
) -> None:
    rows = [
        (
            ticker,
            event_date,
            int(window),
            _FORWARD_RETURN_CACHE_VERSION,
            float(value),
            source,
        )
        for event_date, by_window in values.items()
        for window, value in by_window.items()
        if np.isfinite(_as_float(value))
    ]
    if not rows:
        return
    path = _forward_return_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as con:
            _ensure_forward_return_table(con)
            con.executemany(
                """INSERT INTO forward_return_cache
                   (ticker, event_date, window, version, value, source)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, event_date, window, version) DO UPDATE SET
                       value=excluded.value,
                       source=excluded.source,
                       updated_at=CURRENT_TIMESTAMP""",
                rows,
            )
    except (OSError, sqlite3.Error):
        # Live results remain usable even if the hosting filesystem is
        # read-only or another process temporarily owns the database.
        return


@dataclass(frozen=True)
class CoefficientEstimate:
    """Inference for one coefficient in a fitted model."""

    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def coef(self) -> float:
        """Short alias useful in tables."""
        return self.coefficient

    @property
    def se(self) -> float:
        """Short alias useful in tables."""
        return self.std_error


@dataclass
class RegressionResult:
    # Original fields are kept first (and with their original meaning) so
    # existing app code can continue to use ``coef`` and ``t_stat``.
    window: int
    n: int
    coef: float = np.nan
    intercept: float = np.nan
    t_stat: float = np.nan
    r2: float = np.nan
    rmse: float = np.nan
    mae: float = np.nan
    controls: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Rich inference requested by the project rubric.
    std_error: float = np.nan
    p_value: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    adjusted_r2: float = np.nan
    model: str = "sentiment"
    signal_name: str = "sentiment"
    coefficients: dict[str, CoefficientEstimate] = field(default_factory=dict)
    covariance: str = "classical"
    cluster_by: str | None = None
    n_clusters: int | None = None
    reference_categories: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def adj_r2(self) -> float:
        """Alias for display code that uses the conventional short name."""
        return self.adjusted_r2

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    def summary(self) -> str:
        base = (
            f"{self.window}-day {self.model} model | n={self.n}, "
            f"b={self.coef:.4f} (SE={self.std_error:.4f}, "
            f"t={self.t_stat:.2f}, p={self.p_value:.4g}, "
            f"95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}]), "
            f"R2={self.r2:.3f}, adjusted R2={self.adjusted_r2:.3f}, "
            f"RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"
        )
        if self.cluster_by:
            base += f" | SE clustered by {self.cluster_by} (G={self.n_clusters})"
        return base


def _close_series(frame, ticker: str):
    """Normalize Ticker.history and yf.download output to one Close series."""
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return pd.Series(dtype=float)
    try:
        close = frame["Close"]
    except (KeyError, TypeError):
        return pd.Series(dtype=float)
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        elif len(close.columns):
            close = close.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return close
    index = pd.to_datetime(close.index, errors="coerce")
    valid = ~index.isna()
    close = close.iloc[np.flatnonzero(valid)].copy()
    index = index[valid]
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    return close[~close.index.duplicated(keep="last")].sort_index()


def _bounded_price_history(yf, ticker: str, start, end):
    """Try two short, bounded yfinance routes before declaring failure."""
    errors: list[str] = []
    try:
        frame = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
            timeout=_PRICE_TIMEOUT_SECONDS,
        )
        close = _close_series(frame, ticker)
        if not close.empty:
            return close, "yfinance.history"
        errors.append("Ticker.history returned no rows")
    except Exception as exc:  # yfinance uses several provider-specific errors
        errors.append(f"Ticker.history: {type(exc).__name__}: {exc}")

    # yf.download follows a separate code path and is a useful fallback when a
    # Ticker object has stale cookie/crumb state.  Keep it single-threaded so
    # retries do not multiply requests in a hosted Streamlit process.
    try:
        download = getattr(yf, "download")
        frame = download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
            timeout=_PRICE_TIMEOUT_SECONDS,
        )
        close = _close_series(frame, ticker)
        if not close.empty:
            return close, "yfinance.download"
        errors.append("yf.download returned no rows")
    except Exception as exc:
        errors.append(f"yf.download: {type(exc).__name__}: {exc}")

    # One small delay prevents an immediate tight retry loop at the app level;
    # persistent successes mean normal subsequent calls never reach here.
    time.sleep(0.1)
    return None, "; ".join(errors)


def fetch_forward_returns(ticker: str, dates: list[str], windows=(5, 20)) -> dict:
    """Return forward returns using a bounded request and persistent cache.

    Historical results are immutable for this study definition and are stored
    in ``finsight.db``.  A cache-complete call performs no network request.
    Missing live data raises :class:`MarketDataUnavailable` instead of silently
    producing regressions with ``N=0``.
    """
    import pandas as pd

    ticker = str(ticker).strip().upper()
    unique_dates = list(dict.fromkeys(str(date) for date in dates))
    requested_windows = tuple(sorted({int(window) for window in windows}))
    if not ticker:
        raise ValueError("ticker is required")
    if not unique_dates:
        return {}
    if not requested_windows or any(window < 1 for window in requested_windows):
        raise ValueError("forward-return windows must be positive integers")

    parsed_dates: dict[str, pd.Timestamp] = {}
    for date in unique_dates:
        try:
            parsed_dates[date] = pd.Timestamp(date).tz_localize(None).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid event date {date!r}") from exc

    output = _cached_forward_returns(ticker, unique_dates, requested_windows)
    missing = {
        (date, window)
        for date in unique_dates
        for window in requested_windows
        if window not in output[date]
    }
    if not missing:
        return output

    earliest = min(parsed_dates.values()) - pd.Timedelta(days=7)
    calendar_buffer = max(requested_windows) * 2 + 14
    desired_end = max(parsed_dates.values()) + pd.Timedelta(days=calendar_buffer)
    # Yahoo's end is exclusive.  Do not ask far into the future, which can
    # trigger misleading "possibly delisted" responses in some versions.
    tomorrow = pd.Timestamp(datetime.now(timezone.utc).date()) + pd.Timedelta(days=1)
    request_end = min(desired_end, tomorrow)
    if request_end <= earliest:
        request_end = earliest + pd.Timedelta(days=1)

    import yfinance as yf

    _configure_yfinance_cache(yf)
    close, source = _bounded_price_history(yf, ticker, earliest, request_end)
    if close is None or close.empty:
        cached_count = sum(len(values) for values in output.values())
        if cached_count:
            return output
        raise MarketDataUnavailable(
            f"No market prices are available for {ticker} from "
            f"{earliest.date()} to {request_end.date()}. "
            f"Both bounded yfinance routes failed ({source}). Check outbound "
            "network access or pre-populate the forward-return cache."
        )

    computed = {date: {} for date in unique_dates}
    for date, window in sorted(missing):
        index = close.index.searchsorted(parsed_dates[date], side="right")
        if index + window < len(close):
            p0, p1 = float(close.iloc[index]), float(close.iloc[index + window])
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                value = p1 / p0 - 1.0
                output[date][window] = value
                computed[date][window] = value
    _persist_forward_returns(ticker, computed, source)

    if not any(output.values()):
        latest = close.index.max().date() if len(close) else "unknown"
        raise MarketDataUnavailable(
            f"Prices for {ticker} were retrieved through {latest}, but none of "
            f"the requested {requested_windows}-day returns are complete for "
            f"event dates {min(unique_dates)} through {max(unique_dates)}."
        )
    return output


def _as_float(value: object) -> float:
    try:
        answer = float(value)
    except (TypeError, ValueError):
        return np.nan
    return answer if np.isfinite(answer) else np.nan


def _first_present(row: Mapping[str, object], names: Sequence[str]) -> float:
    for name in names:
        if name in row:
            return _as_float(row[name])
    return np.nan


def _mean_finite(values: Iterable[object]) -> float:
    numbers = np.asarray([_as_float(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(numbers.mean()) if len(numbers) else np.nan


def _mode_text(values: Iterable[object], default: str = "Unknown") -> str:
    cleaned = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not cleaned:
        return default
    counts = Counter(cleaned)
    # Lexical tie breaking makes results independent of input row order.
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _mean_window_maps(rows: Sequence[Mapping[str, object]], key: str) -> dict[int, float]:
    windows: set[int] = set()
    for row in rows:
        value = row.get(key, {})
        if isinstance(value, Mapping):
            for window in value:
                try:
                    windows.add(int(window))
                except (TypeError, ValueError):
                    continue
    result: dict[int, float] = {}
    for window in windows:
        vals = []
        for row in rows:
            mapping = row.get(key, {})
            if isinstance(mapping, Mapping):
                vals.append(mapping.get(window, mapping.get(str(window), np.nan)))
        result[window] = _mean_finite(vals)
    return result


def aggregate_event_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Collapse document rows to one observation per ``ticker`` and ``date``.

    Continuous signals are averaged within an event.  Sector is the modal
    non-empty value.  Multiple document forms are represented by a stable
    ``+``-joined category (for example ``10-Q+TRANSCRIPT``), allowing form
    controls without re-introducing duplicate return observations.

    If callers pass already-enriched ``returns`` and ``market`` mappings they
    are averaged too; this makes the pure regression layer easy to use offline.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        date = str(row.get("date", "")).strip()
        if not ticker or not date:
            continue
        grouped.setdefault((ticker, date), []).append(row)

    events: list[dict] = []
    for (ticker, date), items in sorted(grouped.items()):
        forms = sorted({
            str(item.get("form", "")).strip().upper()
            for item in items
            if str(item.get("form", "")).strip()
        })
        event = {
            "ticker": ticker,
            "date": date,
            "sentiment": _mean_finite(
                _first_present(item, ("sentiment", "signal")) for item in items
            ),
            "disclosure_change": _mean_finite(
                _first_present(
                    item, ("disclosure_change", "change_signal", "disclosure")
                )
                for item in items
            ),
            "sector": _mode_text(
                (item.get("sector") for item in items),
                default=sector_for_ticker(ticker),
            ),
            "form": "+".join(forms) if forms else "Unknown",
            "forms": tuple(forms),
            "document_count": len(items),
            "returns": _mean_window_maps(items, "returns"),
            "market": _mean_window_maps(items, "market"),
        }
        events.append(event)
    return events


def _t_distribution_values(t_stats: np.ndarray, df: int) -> tuple[np.ndarray, float]:
    """Return two-sided p-values and the 95% critical value.

    SciPy is already an indirect project dependency, but a normal fallback
    keeps the core useful in a minimal installation.
    """
    try:
        from scipy.stats import t as student_t

        p_values = 2.0 * student_t.sf(np.abs(t_stats), df=max(int(df), 1))
        critical = float(student_t.ppf(0.975, df=max(int(df), 1)))
    except ImportError:  # pragma: no cover - exercised only in minimal installs
        p_values = np.asarray([
            math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in t_stats
        ])
        critical = 1.959963984540054
    return p_values, critical


def _fit_matrix(
    X: np.ndarray,
    y: np.ndarray,
    term_names: Sequence[str],
    *,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, int, str]:
    """Fit OLS and return beta/inference/fit statistics.

    Clustered covariance uses the standard one-way CR1 finite-sample
    correction ``G/(G-1) * (n-1)/(n-k)`` and Student-t inference with ``G-1``
    degrees of freedom.  Classical OLS uses ``n-k`` degrees of freedom.
    """
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        raise ValueError("regression design is rank deficient or has no residual degrees of freedom")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    bread = np.linalg.inv(X.T @ X)

    covariance_name = "classical OLS"
    if groups is not None:
        labels, inverse = np.unique(groups.astype(str), return_inverse=True)
        g = len(labels)
        if g < 2:
            raise ValueError("cluster-robust covariance requires at least two clusters")
        meat = np.zeros((k, k), dtype=float)
        for group_index in range(g):
            in_group = inverse == group_index
            score = X[in_group].T @ residuals[in_group]
            meat += np.outer(score, score)
        correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
        covariance = correction * bread @ meat @ bread
        inference_df = g - 1
        covariance_name = "cluster-robust CR1"
    else:
        sigma2 = float(residuals @ residuals) / (n - k)
        covariance = sigma2 * bread
        inference_df = n - k

    variances = np.diag(covariance)
    # Tiny negative values can occur from floating-point multiplication.
    variances = np.where((variances < 0) & (variances > -1e-14), 0.0, variances)
    standard_errors = np.sqrt(np.where(variances >= 0, variances, np.nan))
    t_stats = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values, critical = _t_distribution_values(t_stats, inference_df)
    ci_low = beta - critical * standard_errors
    ci_high = beta + critical * standard_errors

    residual_ss = float(residuals @ residuals)
    total_ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (n - 1.0) / (n - k)
        if np.isfinite(r2) and n > k
        else np.nan
    )
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return (
        beta,
        standard_errors,
        t_stats,
        p_values,
        np.column_stack((ci_low, ci_high)),
        float(r2),
        float(adjusted_r2),
        rmse,
        mae,
        inference_df,
        covariance_name,
    )


def _empty_result(
    window: int,
    n: int,
    model: str,
    signal_name: str,
    warning: str,
) -> RegressionResult:
    return RegressionResult(
        window=window,
        n=n,
        model=model,
        signal_name=signal_name,
        warnings=(warning,),
    )


def ols(
    signal: np.ndarray,
    ret: np.ndarray,
    window: int,
    controls: dict[str, np.ndarray] | None = None,
) -> RegressionResult:
    """Backward-compatible classical OLS for a signal and numeric controls."""
    x = np.asarray(signal, float)
    y = np.asarray(ret, float)
    ctrl = {k: np.asarray(v, float) for k, v in (controls or {}).items()}

    if any(len(value) != len(x) for value in [y, *ctrl.values()]):
        raise ValueError("signal, return and control arrays must have equal lengths")
    mask = np.isfinite(x) & np.isfinite(y)
    for value in ctrl.values():
        mask &= np.isfinite(value)
    x, y = x[mask], y[mask]
    ctrl = {name: value[mask] for name, value in ctrl.items()}
    n = len(x)
    names = ["intercept", "sentiment", *ctrl]
    X = np.column_stack([np.ones(n), x, *ctrl.values()]) if n else np.empty((0, len(names)))
    if n <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return _empty_result(
            window, n, "sentiment", "sentiment", "insufficient observations or rank-deficient design"
        )

    beta, se, t_stats, p_values, ci, r2, adj_r2, rmse, mae, _, covariance = (
        _fit_matrix(X, y, names)
    )
    estimates = {
        name: CoefficientEstimate(
            float(beta[i]), float(se[i]), float(t_stats[i]), float(p_values[i]),
            float(ci[i, 0]), float(ci[i, 1]),
        )
        for i, name in enumerate(names)
    }
    ctrl_out = {
        name: (estimates[name].coefficient, estimates[name].t_stat) for name in ctrl
    }
    signal_estimate = estimates["sentiment"]
    return RegressionResult(
        window=window,
        n=n,
        coef=signal_estimate.coefficient,
        intercept=estimates["intercept"].coefficient,
        t_stat=signal_estimate.t_stat,
        r2=r2,
        rmse=rmse,
        mae=mae,
        controls=ctrl_out,
        std_error=signal_estimate.std_error,
        p_value=signal_estimate.p_value,
        ci_low=signal_estimate.ci_low,
        ci_high=signal_estimate.ci_high,
        adjusted_r2=adj_r2,
        model="sentiment",
        signal_name="sentiment",
        coefficients=estimates,
        covariance=covariance,
    )


_MODEL_SIGNALS: dict[str, tuple[str, ...]] = {
    "sentiment": ("sentiment",),
    "disclosure": ("disclosure_change",),
    "joint": ("sentiment", "disclosure_change"),
}


def _dummy_columns(
    values: np.ndarray,
    prefix: str,
) -> tuple[list[str], list[np.ndarray], str | None]:
    categories = sorted(set(values.astype(str)))
    if len(categories) <= 1:
        return [], [], categories[0] if categories else None
    counts = Counter(values.astype(str))
    # The most frequent category is an interpretable, stable reference.
    reference = sorted(categories, key=lambda value: (-counts[value], value))[0]
    included = [value for value in categories if value != reference]
    return (
        [f"{prefix}[{value}]" for value in included],
        [(values.astype(str) == value).astype(float) for value in included],
        reference,
    )


def _select_clusters(
    events: Sequence[Mapping[str, object]],
    mask: np.ndarray,
    cluster_by: str | None,
) -> tuple[np.ndarray | None, str | None, list[str]]:
    if cluster_by is None or str(cluster_by).lower() in {"none", "classical"}:
        return None, None, []
    requested = str(cluster_by).lower().replace("event_date", "date")
    if requested not in {"auto", "ticker", "date"}:
        raise ValueError("cluster_by must be 'auto', 'ticker', 'date', or None")

    ticker = np.asarray([str(event["ticker"]) for event in events], dtype=object)[mask]
    date = np.asarray([str(event["date"]) for event in events], dtype=object)[mask]
    warnings: list[str] = []
    chosen = "ticker" if requested == "auto" else requested
    groups = ticker if chosen == "ticker" else date
    if len(np.unique(groups)) < 2 and chosen == "ticker":
        groups, chosen = date, "date"
        warnings.append(
            "ticker clustering is unidentified with one ticker; used event-date clusters"
        )
    if len(np.unique(groups)) < 2:
        return None, None, warnings + [
            "fewer than two ticker/date clusters; cluster-robust inference unavailable"
        ]
    return groups, chosen, warnings


def _fit_event_model(
    events: Sequence[Mapping[str, object]],
    window: int,
    model: str,
    *,
    market_control: bool,
    sector_control: bool,
    form_control: bool,
    cluster_by: str | None,
) -> RegressionResult:
    if model not in _MODEL_SIGNALS:
        raise ValueError(f"unknown model {model!r}; choose from {tuple(_MODEL_SIGNALS)}")
    signal_names = _MODEL_SIGNALS[model]
    primary_name = signal_names[0]

    y = np.asarray([
        _as_float(event.get("returns", {}).get(window, np.nan))
        if isinstance(event.get("returns"), Mapping) else np.nan
        for event in events
    ])
    numeric = {
        name: np.asarray([_as_float(event.get(name)) for event in events])
        for name in signal_names
    }
    if market_control:
        numeric["mkt"] = np.asarray([
            _as_float(event.get("market", {}).get(window, np.nan))
            if isinstance(event.get("market"), Mapping) else np.nan
            for event in events
        ])

    mask = np.isfinite(y)
    for value in numeric.values():
        mask &= np.isfinite(value)
    n = int(mask.sum())
    # Require residual degrees of freedom for the intercept and currently
    # requested numeric terms.  We check again after dummy expansion.
    minimum_columns = 1 + len(numeric)
    if n <= minimum_columns:
        return _empty_result(
            window,
            n,
            model,
            primary_name,
            "too few complete events for the requested numeric terms",
        )

    names = ["intercept", *numeric]
    columns = [np.ones(n), *(value[mask] for value in numeric.values())]
    references: dict[str, str] = {}

    if sector_control:
        sectors = np.asarray([
            str(event.get("sector") or "Unknown") for event in events
        ], dtype=object)[mask]
        dummy_names, dummy_values, reference = _dummy_columns(sectors, "sector")
        names.extend(dummy_names)
        columns.extend(dummy_values)
        if reference is not None:
            references["sector"] = reference

    # The requested disclosure-only specification omits form.  Sentiment and
    # joint specifications control for systematic baseline differences among
    # 10-K, 10-Q and transcript language.
    if form_control and model in {"sentiment", "joint"}:
        forms = np.asarray([
            str(event.get("form") or "Unknown") for event in events
        ], dtype=object)[mask]
        dummy_names, dummy_values, reference = _dummy_columns(forms, "form")
        names.extend(dummy_names)
        columns.extend(dummy_values)
        if reference is not None:
            references["form"] = reference

    X = np.column_stack(columns)
    if n <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return _empty_result(
            window,
            n,
            model,
            primary_name,
            "insufficient residual degrees of freedom or rank-deficient controls",
        )

    groups, cluster_name, fit_warnings = _select_clusters(events, mask, cluster_by)
    if groups is not None and len(np.unique(groups)) < 10:
        fit_warnings.append(
            "fewer than 10 clusters; cluster-robust p-values and confidence intervals may be unstable"
        )
    try:
        beta, se, t_stats, p_values, ci, r2, adj_r2, rmse, mae, _, covariance = (
            _fit_matrix(X, y[mask], names, groups=groups)
        )
    except ValueError as exc:
        return _empty_result(window, n, model, primary_name, str(exc))

    estimates = {
        name: CoefficientEstimate(
            float(beta[i]), float(se[i]), float(t_stats[i]), float(p_values[i]),
            float(ci[i, 0]), float(ci[i, 1]),
        )
        for i, name in enumerate(names)
    }
    primary = estimates[primary_name]
    controls = {
        name: (estimate.coefficient, estimate.t_stat)
        for name, estimate in estimates.items()
        if name not in {"intercept", *signal_names}
    }
    return RegressionResult(
        window=window,
        n=n,
        coef=primary.coefficient,
        intercept=estimates["intercept"].coefficient,
        t_stat=primary.t_stat,
        r2=r2,
        rmse=rmse,
        mae=mae,
        controls=controls,
        std_error=primary.std_error,
        p_value=primary.p_value,
        ci_low=primary.ci_low,
        ci_high=primary.ci_high,
        adjusted_r2=adj_r2,
        model=model,
        signal_name=primary_name,
        coefficients=estimates,
        covariance=covariance,
        cluster_by=cluster_name,
        n_clusters=len(np.unique(groups)) if groups is not None else None,
        reference_categories=references,
        warnings=tuple(fit_warnings),
    )


def fit_event_regressions(
    rows: Sequence[Mapping[str, object]],
    windows: Sequence[int] = (5, 20),
    *,
    models: Sequence[str] | None = None,
    market_control: bool = True,
    sector_control: bool = True,
    form_control: bool = True,
    cluster_by: str | None = "auto",
) -> list[RegressionResult]:
    """Fit requested models to rows that already contain return mappings.

    ``rows`` may be document-level or event-level; duplicate ticker-date rows
    are always collapsed.  Each row accepts these fields:

    ``ticker``, ``date``, ``sentiment`` (or legacy ``signal``),
    ``disclosure_change``, ``sector``, ``form``, ``returns`` and ``market``.

    ``models=None`` runs the sentiment model and automatically adds disclosure
    and joint models when a disclosure signal is present.  Explicit model names
    are ``sentiment``, ``disclosure`` and ``joint``.
    """
    events = aggregate_event_rows(rows)
    if models is None:
        has_disclosure = any(np.isfinite(_as_float(row.get("disclosure_change"))) for row in events)
        models = ("sentiment", "disclosure", "joint") if has_disclosure else ("sentiment",)
    invalid = [model for model in models if model not in _MODEL_SIGNALS]
    if invalid:
        raise ValueError(f"unknown regression model(s): {', '.join(invalid)}")

    return [
        _fit_event_model(
            events,
            int(window),
            model,
            market_control=market_control,
            sector_control=sector_control,
            form_control=form_control,
            cluster_by=cluster_by,
        )
        for window in windows
        for model in models
    ]


def run_study(
    rows: list[dict],
    windows=(5, 20),
    market_control: bool = True,
    *,
    models: Sequence[str] | None = None,
    sector_control: bool = True,
    form_control: bool = True,
    cluster_by: str | None = "auto",
) -> list[RegressionResult]:
    """Fetch prices and run event-level controlled regression models.

    Price calls happen after ticker-date aggregation, reducing network work and
    ensuring duplicate documents never inflate the regression sample size.
    """
    events = aggregate_event_rows(rows)
    if not events:
        return []

    all_dates = sorted({str(event["date"]) for event in events})
    market = (
        fetch_forward_returns(MARKET_PROXY, all_dates, windows)
        if market_control else {}
    )
    if market_control and not any(bool(values) for values in market.values()):
        raise MarketDataUnavailable(
            f"{MARKET_PROXY} returned no complete forward returns for the "
            "requested event dates; controlled regressions cannot be estimated."
        )

    by_ticker: dict[str, list[dict]] = {}
    for event in events:
        by_ticker.setdefault(str(event["ticker"]), []).append(event)
    enriched: list[dict] = []
    for ticker, items in by_ticker.items():
        dates = sorted({str(item["date"]) for item in items})
        forward = fetch_forward_returns(ticker, dates, windows)
        if not any(bool(values) for values in forward.values()):
            raise MarketDataUnavailable(
                f"{ticker} returned no complete forward returns for its event "
                "dates; refusing to display an empty regression."
            )
        for item in items:
            event = dict(item)
            event["returns"] = forward.get(str(item["date"]), {})
            event["market"] = market.get(str(item["date"]), {})
            enriched.append(event)

    return fit_event_regressions(
        enriched,
        windows,
        models=models,
        market_control=market_control,
        sector_control=sector_control,
        form_control=form_control,
        cluster_by=cluster_by,
    )
