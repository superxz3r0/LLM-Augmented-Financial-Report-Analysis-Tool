from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FINSIGHT_DATA_DIR", PROJECT_ROOT / "data"))
SAMPLE_DIR = DATA_DIR / "sample"
FILINGS_DIR = DATA_DIR / "filings"
INDEX_DIR = DATA_DIR / "index"


@dataclass
class Settings:
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    finbert_model: str = "ProsusAI/finbert"
    substantive_similarity_threshold: float = 0.80
    boilerplate_min_ratio: float = 0.97
    forward_windows: tuple = (5, 20)
    extra: dict = field(default_factory=dict)


settings = Settings()

_embedder_cache: dict[str, object] = {}


def get_embedder(model_name: str | None = None):
    """Lazily load and cache a SentenceTransformer by name.

    Loading the model (bge-large is ~1.3GB) dominates wall-clock time far
    more than actually encoding text with it, so every caller (diff, audit,
    index) shares one instance per process instead of each constructing its
    own — diff.py in particular used to build a fresh instance per Item
    section, reloading the model many times over on a single "Run diff" click.
    """
    name = model_name or settings.embedding_model
    if name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache[name] = SentenceTransformer(name)
    return _embedder_cache[name]
