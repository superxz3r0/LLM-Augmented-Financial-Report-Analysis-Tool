from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from finsight import returns


def _row(
    ticker: str,
    date: str,
    sentiment: float,
    disclosure: float | None,
    market: float,
    forward: float,
    *,
    sector: str | None = None,
    form: str = "10-Q",
) -> dict:
    row = {
        "ticker": ticker,
        "date": date,
        "sentiment": sentiment,
        "disclosure_change": disclosure,
        "form": form,
        "market": {5: market},
        "returns": {5: forward},
    }
    if sector is not None:
        row["sector"] = sector
    return row


def test_event_aggregation_averages_signals_and_does_not_fill_missing_change():
    rows = [
        _row("aapl", "2024-01-10", 0.2, None, 0.01, 0.03, form="10-Q"),
        _row("AAPL", "2024-01-10", 0.6, 0.8, 0.01, 0.03, form="TRANSCRIPT"),
        _row("AAPL", "2024-02-10", -0.1, None, 0.02, 0.01, form="10-K"),
    ]

    events = returns.aggregate_event_rows(rows)

    assert len(events) == 2
    assert events[0]["ticker"] == "AAPL"
    assert events[0]["document_count"] == 2
    assert events[0]["sentiment"] == pytest.approx(0.4)
    assert events[0]["disclosure_change"] == pytest.approx(0.8)
    assert events[0]["form"] == "10-Q+TRANSCRIPT"
    assert events[0]["sector"] == "Information Technology"
    # No prior document means unavailable, not "no change".
    assert np.isnan(events[1]["disclosure_change"])


def test_cluster_cr1_matches_direct_matrix_calculation():
    rng = np.random.default_rng(812)
    rows = []
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PFE", "WMT"]
    for ticker_index, ticker in enumerate(tickers):
        cluster_shock = rng.normal(scale=0.02)
        for observation in range(12):
            sentiment = rng.normal()
            market = rng.normal(scale=0.03)
            noise = cluster_shock + rng.normal(scale=0.01)
            forward = 0.015 + 0.12 * sentiment + 0.7 * market + noise
            rows.append(
                _row(
                    ticker,
                    f"2024-{observation + 1:02d}-{ticker_index + 1:02d}",
                    sentiment,
                    None,
                    market,
                    forward,
                )
            )

    result = returns.fit_event_regressions(
        rows,
        windows=(5,),
        models=("sentiment",),
        sector_control=False,
        form_control=False,
        cluster_by="ticker",
    )[0]

    x = np.asarray([row["sentiment"] for row in rows])
    mkt = np.asarray([row["market"][5] for row in rows])
    y = np.asarray([row["returns"][5] for row in rows])
    X = np.column_stack((np.ones(len(rows)), x, mkt))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((3, 3))
    for ticker in tickers:
        mask = np.asarray([row["ticker"] == ticker for row in rows])
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    n, k, g = len(rows), X.shape[1], len(tickers)
    expected_cov = (
        g / (g - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    )

    assert result.coef == pytest.approx(beta[1])
    assert result.std_error == pytest.approx(np.sqrt(expected_cov[1, 1]))
    assert result.covariance == "cluster-robust CR1"
    assert result.cluster_by == "ticker"
    assert result.n_clusters == len(tickers)
    assert 0 <= result.p_value <= 1
    assert result.ci_low < result.coef < result.ci_high


def test_three_specs_have_requested_controls_and_complete_statistics():
    rng = np.random.default_rng(2025)
    ticker_sector = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "PFE": "Health Care",
        "JNJ": "Health Care",
    }
    rows = []
    for ticker_index, (ticker, sector) in enumerate(ticker_sector.items()):
        company_shock = rng.normal(scale=0.004)
        for observation in range(18):
            sentiment = rng.normal()
            disclosure = rng.uniform()
            market = rng.normal(scale=0.02)
            form = ("10-Q", "10-K", "TRANSCRIPT")[(observation + ticker_index) % 3]
            sector_effect = {
                "Information Technology": 0.006,
                "Financials": -0.003,
                "Energy": 0.004,
                "Health Care": 0.0,
            }[sector]
            form_effect = {"10-Q": -0.002, "10-K": 0.003, "TRANSCRIPT": 0.008}[form]
            forward = (
                0.01
                + 0.035 * sentiment
                + 0.06 * disclosure
                + 0.8 * market
                + sector_effect
                + form_effect
                + company_shock
                + rng.normal(scale=0.01)
            )
            rows.append(
                _row(
                    ticker,
                    f"{2020 + observation // 12}-{observation % 12 + 1:02d}-{ticker_index + 1:02d}",
                    sentiment,
                    disclosure,
                    market,
                    forward,
                    sector=sector,
                    form=form,
                )
            )

    results = returns.fit_event_regressions(
        rows,
        windows=(5,),
        models=("sentiment", "disclosure", "joint"),
    )
    by_model = {result.model: result for result in results}

    assert set(by_model) == {"sentiment", "disclosure", "joint"}
    for result in results:
        assert result.n == len(rows)
        assert np.isfinite(result.std_error)
        assert np.isfinite(result.t_stat)
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.ci_low)
        assert np.isfinite(result.ci_high)
        assert result.adjusted_r2 <= result.r2 + 1e-12
        assert result.reference_categories["sector"] in set(ticker_sector.values())
        assert "mkt" in result.coefficients
        assert any(name.startswith("sector[") for name in result.coefficients)

    assert any(name.startswith("form[") for name in by_model["sentiment"].coefficients)
    assert not any(name.startswith("form[") for name in by_model["disclosure"].coefficients)
    assert any(name.startswith("form[") for name in by_model["joint"].coefficients)
    assert set(("sentiment", "disclosure_change")) <= set(by_model["joint"].coefficients)


def test_one_company_falls_back_to_event_date_clusters():
    rng = np.random.default_rng(91)
    rows = []
    for index in range(30):
        signal = rng.normal()
        market = rng.normal(scale=0.02)
        rows.append(
            _row(
                "AAPL",
                f"{2020 + index // 12}-{index % 12 + 1:02d}-15",
                signal,
                None,
                market,
                0.04 * signal + 0.5 * market + rng.normal(scale=0.01),
                form="10-Q" if index % 2 else "10-K",
            )
        )

    result = returns.fit_event_regressions(
        rows, windows=(5,), models=("sentiment",), cluster_by="ticker"
    )[0]

    assert result.cluster_by == "date"
    assert result.n_clusters == len(rows)
    assert any("used event-date clusters" in warning for warning in result.warnings)


def test_run_study_fetches_prices_only_after_event_deduplication(monkeypatch):
    calls = []

    def fake_returns(ticker, dates, windows):
        calls.append((ticker, tuple(dates)))
        value = 0.01 if ticker == returns.MARKET_PROXY else 0.02
        return {date: {window: value for window in windows} for date in dates}

    monkeypatch.setattr(returns, "fetch_forward_returns", fake_returns)
    rows = []
    for index in range(8):
        date = f"2024-{index + 1:02d}-10"
        rows.append({
            "ticker": "AAPL", "date": date, "signal": index / 10,
            "form": "10-Q",
        })
    # A second document on the first date shares exactly one realised return.
    rows.append({
        "ticker": "AAPL", "date": "2024-01-10", "signal": 0.9,
        "form": "TRANSCRIPT",
    })

    result = returns.run_study(
        rows,
        windows=(5,),
        models=("sentiment",),
        sector_control=False,
        form_control=False,
    )[0]

    assert result.n == 8
    assert calls[0][0] == returns.MARKET_PROXY
    assert calls[1][0] == "AAPL"
    assert len(calls[1][1]) == 8


def test_rank_deficient_model_returns_explanatory_warning():
    rows = [
        _row("AAPL", f"2024-{index + 1:02d}-01", 0.0, None, 0.0, 0.01)
        for index in range(6)
    ]
    result = returns.fit_event_regressions(
        rows,
        windows=(5,),
        models=("sentiment",),
        sector_control=False,
        form_control=False,
    )[0]

    assert np.isnan(result.coef)
    assert result.warnings


def test_forward_returns_use_bounded_range_then_persistent_cache(tmp_path, monkeypatch):
    calls = []
    index = pd.bdate_range("2023-12-20", periods=50)
    prices = pd.DataFrame({"Close": np.linspace(100.0, 125.0, len(index))}, index=index)

    class FakeTicker:
        def __init__(self, ticker):
            assert ticker == "AAPL"

        def history(self, **kwargs):
            calls.append(kwargs)
            return prices

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = FakeTicker
    fake_yf.set_tz_cache_location = lambda _path: None
    fake_yf.download = lambda *_args, **_kwargs: pytest.fail(
        "download fallback should not run when history succeeds"
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(returns, "DATA_DIR", tmp_path)
    monkeypatch.setattr(returns, "_YFINANCE_CACHE_CONFIGURED", False)
    monkeypatch.delenv("FINSIGHT_DB_PATH", raising=False)

    first = returns.fetch_forward_returns("AAPL", ["2024-01-05"], windows=(5,))
    assert 5 in first["2024-01-05"]
    assert len(calls) == 1
    assert "period" not in calls[0]
    assert calls[0]["start"] < "2024-01-05" < calls[0]["end"]

    # A complete cache hit must work with no provider call at all.
    def offline_history(**_kwargs):
        pytest.fail("persistent forward-return cache was not used")

    FakeTicker.history = staticmethod(offline_history)
    second = returns.fetch_forward_returns("AAPL", ["2024-01-05"], windows=(5,))
    assert second == first
    assert (tmp_path / "finsight.db").is_file()


def test_forward_returns_fall_back_to_download(tmp_path, monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=35)
    multi_columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
    downloaded = pd.DataFrame(
        np.linspace(100.0, 115.0, len(index)), index=index, columns=multi_columns
    )
    calls = []

    class EmptyTicker:
        def __init__(self, _ticker):
            pass

        def history(self, **_kwargs):
            calls.append("history")
            return pd.DataFrame()

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = EmptyTicker
    fake_yf.set_tz_cache_location = lambda _path: None

    def fake_download(*_args, **kwargs):
        calls.append(("download", kwargs))
        return downloaded

    fake_yf.download = fake_download
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(returns, "DATA_DIR", tmp_path)
    monkeypatch.setattr(returns, "_YFINANCE_CACHE_CONFIGURED", False)
    monkeypatch.delenv("FINSIGHT_DB_PATH", raising=False)

    result = returns.fetch_forward_returns("AAPL", ["2024-01-05"], windows=(5,))

    assert 5 in result["2024-01-05"]
    assert calls[0] == "history"
    assert calls[1][0] == "download"
    assert calls[1][1]["threads"] is False


def test_empty_live_market_data_raises_instead_of_returning_empty(tmp_path, monkeypatch):
    class EmptyTicker:
        def __init__(self, _ticker):
            pass

        def history(self, **_kwargs):
            return pd.DataFrame()

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = EmptyTicker
    fake_yf.set_tz_cache_location = lambda _path: None
    fake_yf.download = lambda *_args, **_kwargs: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(returns, "DATA_DIR", tmp_path)
    monkeypatch.setattr(returns, "_YFINANCE_CACHE_CONFIGURED", False)
    monkeypatch.delenv("FINSIGHT_DB_PATH", raising=False)
    monkeypatch.setattr(returns.time, "sleep", lambda _seconds: None)

    with pytest.raises(returns.MarketDataUnavailable, match="AAPL.*bounded yfinance"):
        returns.fetch_forward_returns("AAPL", ["2024-01-05"], windows=(5, 20))


def test_run_study_rejects_silent_empty_market_mapping(monkeypatch):
    monkeypatch.setattr(
        returns,
        "fetch_forward_returns",
        lambda _ticker, dates, _windows: {date: {} for date in dates},
    )

    with pytest.raises(returns.MarketDataUnavailable, match="SPY returned no complete"):
        returns.run_study(
            [{"ticker": "AAPL", "date": "2024-01-05", "signal": 0.2}],
            windows=(5,),
        )
