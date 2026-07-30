from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.config import FILINGS_DIR, SAMPLE_DIR
from finsight.extract import extract_signals
from finsight.ingest import load_corpus


def main() -> int:
    labels = {k: v for k, v in
              json.loads((ROOT / "eval" / "labels.json").read_text()).items()
              if not k.startswith("_")}
    docs = {d.doc_id: d for d in load_corpus(FILINGS_DIR, SAMPLE_DIR)}

    per_signal = {"risk_factor_count": [], "revenue_guidance": [],
                  "capex_guidance": [], "sentiment_sign": []}

    for doc_id, truth in labels.items():
        if doc_id not in docs:
            print(f"[skip] {doc_id}: filing not on disk")
            continue
        sig = extract_signals(docs[doc_id])

        #allow 20% Tolerance, cause The most commonly used indicator 
        #for Inter-Annotator Agreement (IAA) is Cohen's Kappa(κ), and when
        #the percentage is higher than 80 pervent will be considered as Strong consistency
        truth_rc = truth["risk_factor_count"]
        tol = max(1, round(truth_rc * 0.20))
        per_signal["risk_factor_count"].append(
            abs(sig.risk_factor_count - truth_rc) <= tol)
        
        extracted_rev = " ".join(sig.revenue_guidance)
        per_signal["revenue_guidance"].append(
            all(x in extracted_rev for x in truth["revenue_guidance_contains"]))
        
        extracted_capex = " ".join(sig.capex_guidance)
        per_signal["capex_guidance"].append(
            all(x in extracted_capex for x in truth["capex_guidance_contains"]))
        
        sign = (sig.sentiment_score > 0) - (sig.sentiment_score < 0)
        per_signal["sentiment_sign"].append(sign == truth["sentiment_sign"])

    print(f"\nExtraction evaluation — {len(labels)} labelled filings\n" + "-" * 48)
    total_hits = total_n = 0
    for name, hits in per_signal.items():
        if not hits:
            continue
        acc = sum(hits) / len(hits)
        total_hits += sum(hits)
        total_n += len(hits)
        print(f"{name:22s} {sum(hits)}/{len(hits)}  ({acc:.0%})")
    overall = total_hits / max(total_n, 1)
    print("-" * 48)
    print(f"{'OVERALL':22s} {total_hits}/{total_n}  ({overall:.0%})  "
          f"{'PASS (>=85%)' if overall >= 0.85 else 'BELOW TARGET'}")
    return 0 if overall >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
