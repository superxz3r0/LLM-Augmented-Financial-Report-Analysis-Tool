from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.chunker import Chunk, chunk_document
from finsight.config import FILINGS_DIR
from finsight.ingest import Document, load_corpus

JSON_OUT = ROOT / "eval" / "rag_questions.json"

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMD": "Advanced Micro Devices",
    "AMZN": "Amazon",
    "BAC": "Bank of America",
    "COST": "Costco",
    "CRM": "Salesforce",
    "CVX": "Chevron",
    "DIS": "Disney",
    "GOOGL": "Alphabet",
    "GS": "Goldman Sachs",
    "INTC": "Intel",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "META": "Meta",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "PEP": "PepsiCo",
    "PFE": "Pfizer",
    "TSLA": "Tesla",
    "UNH": "UnitedHealth Group",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
}

TOPIC_PROFILES = {
    "risk": [
        ("competition", r"\bcompet"),
        ("regulation or legal proceedings", r"\b(regulat|legal|litigation|compliance|law)\b"),
        ("supply chain or manufacturing", r"\b(supply|supplier|inventory|component|manufactur)\b"),
        ("cybersecurity or data risk", r"\b(cyber|security|data breach|privacy)\b"),
        ("customer demand or sales channels", r"\b(demand|customer|consumer|reseller|sales channel)\b"),
        ("liquidity or capital risk", r"\b(liquidity|cash|capital|debt|credit)\b"),
        ("product or platform risk", r"\b(product|service|platform|technology|innovation)\b"),
    ],
    "mda": [
        ("revenue or net sales", r"\b(revenue|net sales)\b"),
        ("margin or profitability", r"\b(margin|profitability|operating income|gross profit)\b"),
        ("liquidity and capital resources", r"\b(liquidity|cash|capital resources|debt|credit)\b"),
        ("expenses or costs", r"\b(expense|cost|spend|spending)\b"),
        ("growth or demand", r"\b(growth|grew|increase|decrease|decline|demand)\b"),
        ("guidance or outlook", r"\b(guidance|outlook|expect|forecast)\b"),
    ],
    "transcript": [
        ("revenue or net sales", r"\b(revenue|net sales)\b"),
        ("guidance or outlook", r"\b(guidance|outlook|expect|forecast)\b"),
        ("margin or profitability", r"\b(margin|profitability|operating income|gross profit)\b"),
        ("growth or demand", r"\b(growth|grew|increase|decrease|decline|demand)\b"),
        ("cash or capital allocation", r"\b(cash|capital allocation|dividend|buyback|repurchase)\b"),
        ("AI, cloud, or data center demand", r"\b(AI|cloud|data center|datacenter)\b"),
    ],
}

QUESTION_BLUEPRINTS = {
    "10k_risk": (
        "According to Item 1A of {company_possessive} {date} 10-K, what does the filing "
        "say about {topic}?"
    ),
    "10q_mda": (
        "In Item 2 MD&A of {company_possessive} {date} 10-Q, what does management report "
        "about {topic}?"
    ),
    "transcript": (
        "In {company_possessive} earnings call transcript dated {date}, what was said "
        "about {topic}?"
    ),
    "comparison": (
        "Comparing Item 1A of {company_possessive} 10-K filings from {old_date} and "
        "{new_date}, what changed or stayed important regarding {topic}?"
    ),
    "unanswerable": (
        "What revenue guidance does {company} provide for fiscal 2030 in the "
        "available filings?"
    ),
}


@dataclass(frozen=True)
class SelectedEvidence:
    doc: Document
    chunk: Chunk
    topic: str
    evidence: str


def main() -> int:
    docs = load_corpus(FILINGS_DIR)
    by_ticker: dict[str, list[Document]] = {}
    for doc in docs:
        by_ticker.setdefault(doc.ticker, []).append(doc)

    cases = []
    qn = 1

    for ticker in sorted(by_ticker):
        picked = pick_latest_with_evidence(by_ticker[ticker], "10-K", "1A", "risk")
        if picked:
            cases.append(make_case(qn, "10k_risk", picked, "single-filing factual",
                                   "medium"))
            qn += 1

    for ticker in sorted(by_ticker):
        picked = pick_latest_with_evidence(by_ticker[ticker], "10-Q", "2", "mda")
        if picked:
            cases.append(make_case(qn, "10q_mda", picked, "section-specific",
                                   "medium"))
            qn += 1

    for ticker in sorted(by_ticker):
        picked = pick_latest_with_evidence(by_ticker[ticker], "TRANSCRIPT", "0",
                                           "transcript")
        if picked:
            cases.append(make_case(qn, "transcript", picked, "transcript QA",
                                   "medium"))
            qn += 1

    for ticker in sorted(by_ticker):
        pair = pick_10k_pair(by_ticker[ticker])
        if pair:
            old, new = pair
            cases.append(make_comparison_case(qn, old, new))
            qn += 1
        if sum(c["question_type"] == "cross-filing comparison" for c in cases) >= 16:
            break

    for ticker in sorted(by_ticker)[:8]:
        cases.append(make_unanswerable_case(qn, ticker))
        qn += 1

    write_outputs(cases)
    print(f"Wrote {len(cases)} RAG evaluation cases to {JSON_OUT}")
    print("Breakdown:")
    for name in sorted({c["question_type"] for c in cases}):
        print(f"  {name}: {sum(c['question_type'] == name for c in cases)}")
    return 0


def write_outputs(cases: list[dict]) -> None:
    JSON_OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def pick_latest_with_evidence(docs: list[Document], form: str,
                              item: str, profile: str) -> SelectedEvidence | None:
    candidates = [d for d in docs if d.form == form and has_item(d, item)]
    for doc in sorted(candidates, key=lambda d: d.date, reverse=True):
        picked = select_evidence(doc, item, profile)
        if picked:
            return picked
    return None


def pick_10k_pair(docs: list[Document]) -> tuple[SelectedEvidence, SelectedEvidence] | None:
    filings = [d for d in docs if d.form == "10-K" and has_item(d, "1A")]
    filings = sorted(filings, key=lambda d: d.date)
    for old_doc, new_doc in zip(filings, filings[1:]):
        old = select_evidence(old_doc, "1A", "risk",
                              preferred_topic="regulation or legal proceedings")
        new = select_evidence(new_doc, "1A", "risk",
                              preferred_topic="regulation or legal proceedings")
        if old and new:
            return old, new
    if len(filings) >= 2:
        old = select_evidence(filings[-2], "1A", "risk")
        new = select_evidence(filings[-1], "1A", "risk")
        if old and new:
            return old, new
    return None


def has_item(doc: Document, item: str) -> bool:
    return any(s.item == item for s in doc.sections)


def select_evidence(doc: Document, item: str, profile: str,
                    preferred_topic: str | None = None) -> SelectedEvidence | None:
    chunks = [c for c in chunk_document(doc) if c.item == item]
    if not chunks:
        return None

    topics = TOPIC_PROFILES[profile]
    if preferred_topic:
        topics = sorted(topics, key=lambda t: 0 if t[0] == preferred_topic else 1)

    scored: list[tuple[int, str, Chunk, str]] = []
    for chunk in chunks:
        text = clean_for_json(chunk.text)
        if len(text) < 180:
            continue
        for topic, pattern in topics:
            if re.search(pattern, text, flags=re.IGNORECASE):
                evidence = concise_evidence(text, pattern)
                if (180 <= len(evidence) <= 1100
                        and re.search(pattern, evidence, flags=re.IGNORECASE)):
                    score = topic_score(topic, text, profile)
                    scored.append((score, topic, chunk, evidence))
                break

    if not scored:
        chunk = max(chunks, key=lambda c: len(c.text))
        text = clean_for_json(chunk.text)
        evidence = concise_evidence(text, r".")
        if len(evidence) >= 180:
            return SelectedEvidence(doc, chunk, "the filing disclosure", evidence)
        return None

    scored.sort(key=lambda row: (-row[0], len(row[3])))
    _, topic, chunk, evidence = scored[0]
    return SelectedEvidence(doc, chunk, topic, evidence)


def topic_score(topic: str, text: str, profile: str) -> int:
    topic_order = {
        "revenue or net sales": 10,
        "margin or profitability": 9,
        "customer demand or sales channels": 8,
        "growth or demand": 8,
        "liquidity and capital resources": 7,
        "liquidity or capital risk": 7,
        "regulation or legal proceedings": 6,
        "supply chain or manufacturing": 5,
        "competition": 4,
        "expenses or costs": 4,
        "guidance or outlook": 3,
        "cash or capital allocation": 3,
        "AI, cloud, or data center demand": 3,
        "product or platform risk": 2,
        "cybersecurity or data risk": 1,
    }
    score = topic_order.get(topic, 0) + min(len(text) // 200, 5)
    if profile in {"mda", "transcript"} and has_financial_value(text):
        score += 8
    if looks_like_boilerplate(text):
        score -= 20
    return score


def concise_evidence(text: str, pattern: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text[:900].strip()
    match_index = 0
    for i, sentence in enumerate(sentences):
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            match_index = i
            break
    start = match_index
    picked: list[str] = []
    total = 0
    for sentence in sentences[start:]:
        if total + len(sentence) > 900 and picked:
            break
        picked.append(sentence)
        total += len(sentence) + 1
        if total >= 240 and len(picked) >= 2:
            break
    return " ".join(picked).strip()


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [s.strip() for s in raw if 60 <= len(s.strip()) <= 1200]


def clean_for_json(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\b[A-Za-z&., ]+\|\s*\d{4}\s+Form\s+10-[KQ]\s+\|\s+\d+\b",
                  " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_financial_value(text: str) -> bool:
    return bool(re.search(r"(\$[\d,.]+|\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s+"
                          r"(?:million|billion|trillion)\b)", text,
                          flags=re.IGNORECASE))


def looks_like_boilerplate(text: str) -> bool:
    low = text.lower()
    return (
        "table of contents" in low
        or "references to website urls" in low
        or "inactive textual references" in low
        or "forward-looking statements" in low
    )


def make_case(qn: int, kind: str, selected: SelectedEvidence,
              question_type: str, difficulty: str) -> dict:
    doc, chunk = selected.doc, selected.chunk
    company = COMPANY_NAMES.get(doc.ticker, doc.ticker)
    question = QUESTION_BLUEPRINTS[kind].format(
        company=company,
        company_possessive=possessive(company),
        date=doc.date,
        topic=selected.topic,
    )
    return {
        "question_id": f"RAG_Q{qn:03d}",
        "question": question,
        "question_type": question_type,
        "difficulty": difficulty,
        "ticker": doc.ticker,
        "company": company,
        "filing_type": doc.form,
        "filing_date": doc.date,
        "target_section": chunk.section_title,
        "target_item": chunk.item,
        "metric_focus": ["retrieval", "citation", "answer", "faithfulness"],
        "gold_answer": selected.evidence,
        "gold_sources": [source_dict(selected)],
    }


def make_comparison_case(qn: int, old: SelectedEvidence,
                         new: SelectedEvidence) -> dict:
    ticker = new.doc.ticker
    company = COMPANY_NAMES.get(ticker, ticker)
    topic = new.topic if new.topic == old.topic else f"{old.topic} and {new.topic}"
    question = QUESTION_BLUEPRINTS["comparison"].format(
        company=company,
        company_possessive=possessive(company),
        old_date=old.doc.date,
        new_date=new.doc.date,
        topic=topic,
    )
    return {
        "question_id": f"RAG_Q{qn:03d}",
        "question": question,
        "question_type": "cross-filing comparison",
        "difficulty": "hard",
        "ticker": ticker,
        "company": company,
        "filing_type": "10-K",
        "filing_date": f"{old.doc.date} vs {new.doc.date}",
        "target_section": "Risk Factors",
        "target_item": "1A",
        "metric_focus": ["multi-document retrieval", "citation", "faithfulness"],
        "gold_answer": (
            f"In the {old.doc.date} 10-K: {old.evidence} "
            f"In the {new.doc.date} 10-K: {new.evidence}"
        ),
        "gold_sources": [source_dict(old), source_dict(new)],
    }


def make_unanswerable_case(qn: int, ticker: str) -> dict:
    company = COMPANY_NAMES.get(ticker, ticker)
    return {
        "question_id": f"RAG_Q{qn:03d}",
        "question": QUESTION_BLUEPRINTS["unanswerable"].format(company=company),
        "question_type": "unanswerable",
        "difficulty": "hard",
        "ticker": ticker,
        "company": company,
        "filing_type": "mixed",
        "filing_date": "not present in corpus",
        "target_section": "N/A",
        "target_item": "N/A",
        "expected_unanswerable": True,
        "metric_focus": ["abstention", "hallucination control"],
        "gold_answer": (
            f"The available filings do not provide fiscal 2030 revenue guidance "
            f"for {company}."
        ),
        "gold_sources": [],
    }


def source_dict(selected: SelectedEvidence) -> dict:
    doc, chunk = selected.doc, selected.chunk
    return {
        "ticker": doc.ticker,
        "company": COMPANY_NAMES.get(doc.ticker, doc.ticker),
        "form": doc.form,
        "date": doc.date,
        "item": chunk.item,
        "section": chunk.section_title,
        "source_file": doc.path.name,
        "doc_id": doc.doc_id,
        "chunk_id": chunk.chunk_id,
        "acceptable_chunk_ids": acceptable_chunk_ids(chunk.chunk_id),
        "evidence": selected.evidence,
    }


def acceptable_chunk_ids(chunk_id: str) -> list[str]:
    match = re.match(r"^(?P<doc>.+)#(?P<n>\d+)$", chunk_id)
    if not match:
        return [chunk_id]
    doc_id = match.group("doc")
    n = int(match.group("n"))
    ids = []
    for offset in (-1, 0, 1):
        neighbor = n + offset
        if neighbor >= 0:
            ids.append(f"{doc_id}#{neighbor}")
    return ids


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


if __name__ == "__main__":
    raise SystemExit(main())
