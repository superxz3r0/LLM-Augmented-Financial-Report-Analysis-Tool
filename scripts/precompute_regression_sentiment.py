"""Benchmark and precompute the Signal -> returns FinBERT cache.

This utility intentionally scores only real filings (``data/filings``), which
is the same corpus used by the Streamlit Signal -> returns tab.  Results are
written through ``finsight.store`` so cache keys and invalidation rules stay
identical to the app.

The local ProsusAI/finbert cache may contain an older ``pytorch_model.bin``
plus a converted ``model.safetensors`` snapshot.  Loading the safetensors
state directly keeps the precomputation on the GPU without relying on unsafe
pickle deserialisation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _append_extra_site_packages() -> None:
    """Allow reuse of Transformers from another local environment.

    The GPU Python environment is expected to provide PyTorch.  Appending
    (rather than prepending) this path ensures that environment's CUDA-enabled
    PyTorch wins if the extra site-packages directory also contains torch.
    """
    extra = os.environ.get("FINSIGHT_TRANSFORMERS_SITE_PACKAGES")
    if extra and extra not in sys.path:
        sys.path.append(extra)


def document_text_prefix(doc, limit: int) -> str:
    """Match ``doc.full_text[:limit]`` without joining unused trailing text."""
    if limit <= 0:
        return ""
    parts: list[str] = []
    remaining = limit
    for index, section in enumerate(doc.sections):
        piece = ("\n\n" if index else "") + section.text
        parts.append(piece[:remaining])
        remaining -= len(parts[-1])
        if remaining <= 0:
            break
    return "".join(parts)


def _cached_finbert_files() -> tuple[Path, Path]:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--ProsusAI--finbert"
    )
    main_ref = cache_root / "refs" / "main"
    if not main_ref.is_file():
        raise FileNotFoundError(f"FinBERT main cache reference not found: {main_ref}")

    main_snapshot = cache_root / "snapshots" / main_ref.read_text().strip()
    if not (main_snapshot / "config.json").is_file():
        raise FileNotFoundError(f"FinBERT config not found: {main_snapshot}")

    candidates = [
        path
        for path in (cache_root / "snapshots").glob("*/model.safetensors")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            "No cached FinBERT model.safetensors was found. "
            "Download a safetensors revision before running this utility."
        )
    weights = max(candidates, key=lambda path: path.stat().st_size)
    return main_snapshot, weights


def load_gpu_pipeline():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in PyTorch {torch.__version__}; use a CUDA-enabled environment"
        )

    _append_extra_site_packages()
    from safetensors.torch import load_file
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    main_snapshot, weights = _cached_finbert_files()
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(main_snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(main_snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    state = load_file(str(weights), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    # Older BERT checkpoints persisted this deterministic buffer; current
    # Transformers recreates it at runtime instead of registering it in the
    # state dict.  No learned parameter is allowed to be missing or extra.
    allowed_unexpected = {"bert.embeddings.position_ids"}
    unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"FinBERT state mismatch: {incompatible}")
    del state
    model = model.half().to("cuda").eval()
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    metadata = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weights": str(weights),
        "model_load_seconds": round(load_seconds, 3),
    }
    return classifier, metadata


def prepare_documents():
    from finsight.config import FILINGS_DIR
    from finsight.ingest import load_corpus
    from finsight.store import REGRESSION_TEXT_CHARS

    docs = load_corpus(FILINGS_DIR)
    items = [
        (doc.doc_id, document_text_prefix(doc, REGRESSION_TEXT_CHARS))
        for doc in docs
    ]
    doc_ids = [doc_id for doc_id, _text in items]
    if len(set(doc_ids)) != len(doc_ids):
        raise RuntimeError("Duplicate document IDs found in the real filing corpus")
    return docs, items


def sentence_groups(items: list[tuple[str, str]]) -> list[list[str]]:
    from finsight.sentiment import _sentences
    from finsight.store import REGRESSION_MAX_SENTENCES

    return [_sentences(text, REGRESSION_MAX_SENTENCES) for _doc_id, text in items]


def benchmark(
    classifier,
    groups: list[list[str]],
    *,
    sample_size: int,
    batch_size: int,
    inference_chunk_size: int,
) -> dict:
    import torch

    sentences = [sentence for group in groups for sentence in group]
    total = len(sentences)
    count = min(sample_size, total)
    if count == 0:
        raise RuntimeError("No scoreable sentences were found")

    # Evenly span the corpus so the benchmark includes different issuers,
    # filing types and sentence lengths rather than only the first documents.
    sampled = [sentences[(index * total) // count] for index in range(count)]
    warmup = sampled[: min(64, count)]
    classifier(warmup, batch_size=batch_size)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for start in range(0, count, inference_chunk_size):
        classifier(sampled[start : start + inference_chunk_size], batch_size=batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rate = count / elapsed
    return {
        "documents": len(groups),
        "total_sentences": total,
        "benchmark_sentences": count,
        "batch_size": batch_size,
        "inference_chunk_size": inference_chunk_size,
        "benchmark_seconds": round(elapsed, 3),
        "sentences_per_second": round(rate, 3),
        "estimated_inference_seconds": round(total / rate, 1),
    }


def precompute(
    classifier,
    items: list[tuple[str, str]],
    *,
    document_batch_size: int,
) -> dict:
    from finsight import sentiment
    from finsight.store import score_regression_sentiments_cached

    sentiment._FinBert._pipe = classifier
    total_docs = len(items)
    total_hits = 0
    total_computed = 0
    started = time.perf_counter()

    for start in range(0, total_docs, document_batch_size):
        stop = min(start + document_batch_size, total_docs)
        batch = items[start:stop]
        results, hits, computed = score_regression_sentiments_cached(batch)
        bad = [result.backend for result in results if result.backend != "finbert"]
        if bad:
            raise RuntimeError(
                f"Batch {start + 1}-{stop} did not use FinBERT throughout: {sorted(set(bad))}"
            )
        total_hits += hits
        total_computed += computed
        elapsed = time.perf_counter() - started
        rate = stop / elapsed
        eta = (total_docs - stop) / rate if rate else 0.0
        print(
            json.dumps(
                {
                    "completed_documents": stop,
                    "total_documents": total_docs,
                    "cache_hits": total_hits,
                    "computed": total_computed,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1),
                }
            ),
            flush=True,
        )

    return {
        "documents": total_docs,
        "cache_hits": total_hits,
        "computed": total_computed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-sentences", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inference-chunk-size", type=int, default=512)
    parser.add_argument("--document-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.benchmark_sentences,
        args.batch_size,
        args.inference_chunk_size,
        args.document_batch_size,
    ) <= 0:
        raise SystemExit("All size arguments must be positive")

    docs, items = prepare_documents()
    groups = sentence_groups(items)
    print(
        json.dumps(
            {
                "stage": "corpus",
                "documents": len(docs),
                "sentences": sum(len(group) for group in groups),
            }
        ),
        flush=True,
    )
    classifier, model_metadata = load_gpu_pipeline()
    print(json.dumps({"stage": "model", **model_metadata}), flush=True)

    benchmark_result = benchmark(
        classifier,
        groups,
        sample_size=args.benchmark_sentences,
        batch_size=args.batch_size,
        inference_chunk_size=args.inference_chunk_size,
    )
    print(json.dumps({"stage": "benchmark", **benchmark_result}), flush=True)
    if args.benchmark_only:
        return

    result = precompute(
        classifier,
        items,
        document_batch_size=args.document_batch_size,
    )
    print(json.dumps({"stage": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
