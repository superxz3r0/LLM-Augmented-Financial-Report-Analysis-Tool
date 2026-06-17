from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fundamentals:
    period: str
    revenue: float | None = None
    cost_of_revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    total_assets: float | None = None
    total_equity: float | None = None
    total_debt: float | None = None
    cash: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    operating_cash_flow: float | None = None
    capex: float | None = None
    interest_expense: float | None = None


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_ratios(f: Fundamentals) -> dict[str, float | None]:
    gross = None
    if f.revenue is not None and f.cost_of_revenue is not None:
        gross = f.revenue - f.cost_of_revenue
    fcf = None
    if f.operating_cash_flow is not None and f.capex is not None:
        fcf = f.operating_cash_flow - abs(f.capex)

    return {
        "gross_margin": _div(gross, f.revenue),
        "operating_margin": _div(f.operating_income, f.revenue),
        "net_margin": _div(f.net_income, f.revenue),
        "current_ratio": _div(f.current_assets, f.current_liabilities),
        "debt_to_equity": _div(f.total_debt, f.total_equity),
        "roe": _div(f.net_income, f.total_equity),
        "roa": _div(f.net_income, f.total_assets),
        "asset_turnover": _div(f.revenue, f.total_assets),
        "fcf": fcf,
        "fcf_margin": _div(fcf, f.revenue),
        "interest_coverage": _div(f.operating_income, f.interest_expense),
    }


def growth_analysis(periods: list[Fundamentals]) -> dict[str, list[float | None]]:
    items = ("revenue", "operating_income", "net_income", "operating_cash_flow")
    out: dict[str, list[float | None]] = {k: [] for k in items}
    for prev, cur in zip(periods, periods[1:]):
        for k in items:
            a, b = getattr(prev, k), getattr(cur, k)
            out[k].append(b / a - 1 if (a and b is not None and a > 0) else None)
    return out


@dataclass
class Flag:
    severity: str
    metric: str
    message: str
    rule: str


def health_check(periods: list[Fundamentals]) -> list[Flag]:
    flags: list[Flag] = []
    if not periods:
        return flags
    latest = periods[-1]
    r = compute_ratios(latest)
    g = growth_analysis(periods) if len(periods) >= 2 else {}

    def add(sev, metric, msg, rule):
        flags.append(Flag(sev, metric, msg, rule))

    cr = r["current_ratio"]
    if cr is not None:
        if cr < 1.0:
            add("red", "current_ratio",
                f"Current ratio {cr:.2f} — current liabilities exceed current assets.",
                "current_ratio < 1.0")
        elif cr < 1.2:
            add("amber", "current_ratio", f"Current ratio {cr:.2f} is thin.",
                "1.0 <= current_ratio < 1.2")
        else:
            add("green", "current_ratio", f"Current ratio {cr:.2f}.", "current_ratio >= 1.2")

    de = r["debt_to_equity"]
    if de is not None:
        if de > 2.0:
            add("red", "debt_to_equity", f"Debt/equity {de:.2f} — high leverage.",
                "debt_to_equity > 2.0")
        elif de > 1.0:
            add("amber", "debt_to_equity", f"Debt/equity {de:.2f}.", "1.0 < debt_to_equity <= 2.0")
        else:
            add("green", "debt_to_equity", f"Debt/equity {de:.2f}.", "debt_to_equity <= 1.0")

    ic = r["interest_coverage"]
    if ic is not None and ic < 3.0:
        add("red" if ic < 1.5 else "amber", "interest_coverage",
            f"Operating income covers interest only {ic:.1f}x.", "interest_coverage < 3.0")

    fcf = r["fcf"]
    if fcf is not None:
        if fcf < 0:
            add("red", "fcf", "Negative free cash flow in the latest period.", "fcf < 0")
        else:
            add("green", "fcf", "Positive free cash flow.", "fcf >= 0")

    if latest.net_income is not None and latest.operating_cash_flow is not None \
            and latest.net_income > 0 and latest.operating_cash_flow < 0.5 * latest.net_income:
        add("amber", "earnings_quality",
            "Operating cash flow is well below net income — earnings may not be converting to cash.",
            "ocf < 0.5 * net_income while net_income > 0")

    rev_g = g.get("revenue", [])
    if len(rev_g) >= 2 and all(x is not None and x < 0 for x in rev_g[-2:]):
        add("red", "revenue_trend", "Revenue declined two consecutive periods.",
            "revenue YoY < 0 twice in a row")

    margins_ok = [compute_ratios(p)["operating_margin"] for p in periods[-3:]]
    margins_ok = [m for m in margins_ok if m is not None]
    if len(margins_ok) == 3 and margins_ok[0] > margins_ok[1] > margins_ok[2]:
        add("amber", "margin_trend",
            "Operating margin compressed for two consecutive periods.",
            "operating_margin strictly decreasing over 3 periods")

    return flags


def from_yfinance(ticker: str, max_periods: int = 4) -> list[Fundamentals]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow

    def pick(df, *names):
        for n in names:
            if n in df.index:
                return df.loc[n]
        return None

    rows = {
        "revenue": pick(inc, "Total Revenue"),
        "cost_of_revenue": pick(inc, "Cost Of Revenue"),
        "operating_income": pick(inc, "Operating Income"),
        "net_income": pick(inc, "Net Income"),
        "interest_expense": pick(inc, "Interest Expense"),
        "total_assets": pick(bal, "Total Assets"),
        "total_equity": pick(bal, "Stockholders Equity", "Total Equity Gross Minority Interest"),
        "total_debt": pick(bal, "Total Debt"),
        "cash": pick(bal, "Cash And Cash Equivalents"),
        "current_assets": pick(bal, "Current Assets"),
        "current_liabilities": pick(bal, "Current Liabilities"),
        "operating_cash_flow": pick(cf, "Operating Cash Flow"),
        "capex": pick(cf, "Capital Expenditure"),
    }

    cols = list(inc.columns)[:max_periods][::-1]
    out = []
    for col in cols:
        kwargs = {}
        for k, series in rows.items():
            v = None
            if series is not None and col in series.index:
                raw = series[col]
                v = float(raw) if raw == raw else None
            kwargs[k] = v
        out.append(Fundamentals(period=str(col.year), **kwargs))
    return out
