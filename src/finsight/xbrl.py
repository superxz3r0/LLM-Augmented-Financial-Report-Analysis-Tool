from __future__ import annotations

import json
import os
import urllib.request

from .metrics import Fundamentals

TAG_MAP: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfSales"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_equity": ["StockholdersEquity",
                     "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
}

_UA = os.environ.get("EDGAR_IDENTITY", "FinSight student@ucdconnect.ie")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def ticker_to_cik(ticker: str) -> int:
    data = _get_json("https://www.sec.gov/files/company_tickers.json")
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise ValueError(f"Ticker not found on EDGAR: {ticker}")


def parse_companyfacts(facts: dict, max_periods: int = 4) -> list[Fundamentals]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    per_field: dict[str, dict[int, float]] = {}

    for field, candidates in TAG_MAP.items():
        for tag in candidates:
            units = gaap.get(tag, {}).get("units", {})
            entries = units.get("USD") or []
            by_fy: dict[int, tuple[str, float]] = {}
            for e in entries:
                if e.get("form") != "10-K" or e.get("fp") != "FY":
                    continue
                fy, val, filed = e.get("fy"), e.get("val"), e.get("filed", "")
                if fy is None or val is None:
                    continue
                if fy not in by_fy or filed > by_fy[fy][0]:
                    by_fy[fy] = (filed, float(val))
            if by_fy:
                per_field[field] = {fy: v for fy, (_, v) in by_fy.items()}
                break

    years = sorted({fy for vals in per_field.values() for fy in vals})[-max_periods:]
    out = []
    for fy in years:
        kwargs = {f: per_field.get(f, {}).get(fy) for f in TAG_MAP}
        out.append(Fundamentals(period=f"FY{fy}", **kwargs))
    return out


def fetch_fundamentals(ticker: str, max_periods: int = 4) -> list[Fundamentals]:
    cik = ticker_to_cik(ticker)
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    return parse_companyfacts(facts, max_periods=max_periods)
