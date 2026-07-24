"""FinBERT sentence scorer — runs on the ephemeral Verda GPU instance.

    python3 score_sentiment_gpu.py texts.json scores.json

texts.json   {"model": "<hf model id>",
              "docs": [{"id": "<key>", "sentences": ["…", …]}, …]}
scores.json  {"<key>": {"score": <mean pos−neg>, "n_sentences": N}}

Sentence splitting happens on the CALLER side (scripts/precompute.py
imports the app's own finsight/sentiment.py splitter), and the model
name is passed in, so this worker cannot drift from the CPU path in
preprocessing or checkpoint. Score definition mirrors sentiment.py:
mean over sentences of P(positive) − P(negative).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH = 256


def main(in_path: str, out_path: str) -> None:
    cfg = json.load(open(in_path))
    model_name = cfg.get("model") or "ProsusAI/finbert"
    docs = cfg["docs"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # label order differs across FinBERT releases — resolve from the config
    id2label = {i: lab.lower() for i, lab in model.config.id2label.items()}
    pos = next(i for i, lab in id2label.items() if "pos" in lab)
    neg = next(i for i, lab in id2label.items() if "neg" in lab)

    # Flatten every sentence of every doc into ONE stream of uniform
    # batches — this is where the GPU speedup comes from, not per-doc
    # serial calls.
    sents: list[str] = []
    owners: list[str] = []
    for d in docs:
        for s in d["sentences"]:
            sents.append(s)
            owners.append(d["id"])

    # Sort by length so each batch pads to similar lengths — 2-3x faster
    # on full-text corpora. Order is irrelevant: per-doc score is a mean.
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    sents = [sents[i] for i in order]
    owners = [owners[i] for i in order]

    per: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(sents), BATCH):
            enc = tok(sents[i:i + BATCH], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            probs = model(**enc).logits.float().softmax(-1)
            per.extend((probs[:, pos] - probs[:, neg]).cpu().tolist())

    agg: dict[str, list[float]] = defaultdict(list)
    for owner, s in zip(owners, per):
        agg[owner].append(s)

    out = {}
    for d in docs:
        v = agg.get(d["id"], [])
        out[d["id"]] = {"score": (sum(v) / len(v)) if v else 0.0,
                        "n_sentences": len(v)}
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"scored {len(docs)} docs / {len(sents)} sentences on {device}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
