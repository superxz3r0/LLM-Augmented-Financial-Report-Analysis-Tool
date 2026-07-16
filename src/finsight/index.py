"""Retrieval: hybrid lexical + dense index.

Design decision (deliberate departure from the suggested FAISS/Chroma-only
stack): financial queries are keyword-heavy — tickers, "Item 1A", "deferred
revenue" — where exact-term matching (BM25) outperforms embeddings, while
embeddings win on paraphrase ("supply problems" -> "component shortages").
So we run both and merge with Reciprocal Rank Fusion (RRF), the standard
zero-tuning fusion method.

  lexical:  _BM25            — pure-python Okapi BM25, no dependencies
  dense:    _ChromaIndex     — sentence-transformers + ChromaDB (persisted)
            _TfidfIndex      — scikit-learn fallback when no model available
  fusion:   HybridIndex      — RRF over both ranked lists

All backends expose .search(query, k, ticker) -> list[(Chunk, score)], which
is the only interface the RAG layer depends on.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import Chunk
from .config import settings, INDEX_DIR

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")
_FILTER_FIELDS = {"form", "date", "item", "doc_id"}
INDEX_SCHEMA_VERSION = 1
INDEX_METADATA_SCHEMA_VERSION = 2
INDEX_MANIFEST = "index_manifest.json"


class IndexLoadError(RuntimeError):
    """Raised when a required persistent vector index cannot be reused."""


@dataclass
class IndexBuildInfo:
    backend: str
    index_path: str
    index_status: str
    index_reused: bool
    embeddings_regenerated: bool
    rebuild_requested: bool
    rebuild_performed: bool
    rebuild_reason: str
    manifest_version: int | None
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int | None
    chunk_size: int
    chunk_overlap: int
    doc_count: int | None
    chunk_count: int
    corpus_fingerprint: str
    load_time_seconds: float
    query_encoder_loaded: bool = False
    query_encoder_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _searchable_text(chunk: Chunk) -> str:
    """Include filing metadata in retrieval text without changing citations."""
    item = f"Item {chunk.item}" if chunk.item else ""
    return (
        f"{chunk.ticker} {chunk.form} {chunk.date} {item} "
        f"{chunk.section_title}\n{chunk.text}"
    )


def _as_filter_values(value) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    return {str(v).lower() for v in values if v is not None and str(v) != ""}


def _matches_filters(chunk: Chunk, ticker: str | None = None, **filters) -> bool:
    if ticker and chunk.ticker.lower() != ticker.lower():
        return False
    for field, value in filters.items():
        if field not in _FILTER_FIELDS:
            continue
        allowed = _as_filter_values(value)
        if allowed and str(getattr(chunk, field, "")).lower() not in allowed:
            return False
    return True


def index_manifest_path(index_dir: Path | None = None) -> Path:
    return Path(index_dir or INDEX_DIR) / INDEX_MANIFEST


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Lightweight deterministic fingerprint for reuse checks.

    The index should not hash every large filing on each eval run. Chunk ids,
    metadata, and text lengths are enough to detect material corpus/chunking
    changes while keeping startup cheap.
    """
    hasher = hashlib.sha256()
    hasher.update(f"index-schema:{INDEX_SCHEMA_VERSION}|metadata:{INDEX_METADATA_SCHEMA_VERSION}".encode())
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        parts = (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.ticker,
            chunk.form,
            chunk.date,
            chunk.item,
            chunk.section_title,
            str(len(chunk.text)),
        )
        hasher.update("|".join(parts).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:24]


def _current_manifest(
    chunks: list[Chunk],
    *,
    doc_count: int | None = None,
    embedding_dimension: int | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "metadata_schema_version": INDEX_METADATA_SCHEMA_VERSION,
        "embedding_provider": "sentence-transformers",
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "doc_count": doc_count,
        "chunk_count": len(chunks),
        "corpus_fingerprint": fingerprint or corpus_fingerprint(chunks),
        "collection_name": "filings",
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexLoadError(f"index manifest is unreadable: {exc}") from exc


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_mismatch(manifest: dict[str, Any], current: dict[str, Any]) -> str:
    required = (
        "index_schema_version",
        "metadata_schema_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "chunk_size",
        "chunk_overlap",
        "doc_count",
        "chunk_count",
        "corpus_fingerprint",
        "collection_name",
    )
    for key in required:
        if key not in manifest:
            return f"manifest missing {key}"
        if current.get(key) is not None and manifest.get(key) != current.get(key):
            return f"{key} changed from {manifest.get(key)!r} to {current.get(key)!r}"
    return ""


class _BM25:
    """Okapi BM25 (k1=1.5, b=0.75). ~50 lines."""

    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_tokens = [_tokenize(_searchable_text(c)) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)

        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        n = len(chunks)
        self.idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}

    def search(self, query: str, k: int, ticker: str | None = None, **filters):
        q = _tokenize(query)
        scores = []
        for i, counts in enumerate(self.tf):
            if not _matches_filters(self.chunks[i], ticker=ticker, **filters):
                continue
            s = 0.0
            norm = self.K1 * (1 - self.B + self.B * self.doc_len[i] / self.avg_len)
            for t in q:
                f = counts.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.K1 + 1) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return [(self.chunks[i], s) for i, s in scores[:k]]

    def filtered_chunks(self, ticker: str | None = None, **filters) -> list[Chunk]:
        return [c for c in self.chunks if _matches_filters(c, ticker=ticker, **filters)]


class HybridIndex:
    """Reciprocal Rank Fusion over lexical and dense rankings:
    rrf(d) = sum over rankers of 1 / (60 + rank_d). Rank-based, so the two
    incomparable score scales never need calibrating."""

    RRF_K = 60

    def __init__(self, lexical, dense, dense_name: str):
        self.lexical, self.dense = lexical, dense
        self.name = f"hybrid(bm25+{dense_name})"

    def search(self, query: str, k: int, ticker: str | None = None, **filters):
        pool = max(k * 2, 10)
        ranked_lists = [
            self.lexical.search(query, pool, ticker=ticker, **filters),
            self.dense.search(query, pool, ticker=ticker, **filters),
        ]
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for hits in ranked_lists:
            for rank, (chunk, _) in enumerate(hits):
                by_id[chunk.chunk_id] = chunk
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) \
                    + 1.0 / (self.RRF_K + rank + 1)
        order = sorted(fused.items(), key=lambda x: -x[1])
        return [(by_id[cid], score) for cid, score in order[:k]]

    def filtered_chunks(self, ticker: str | None = None, **filters) -> list[Chunk]:
        source = getattr(self.lexical, "filtered_chunks", None)
        if source is not None:
            return source(ticker=ticker, **filters)
        chunks = getattr(self.lexical, "chunks", [])
        return [c for c in chunks if _matches_filters(c, ticker=ticker, **filters)]


class _LegacyChromaIndex:
    def __init__(self, chunks: list[Chunk]):
        import hashlib

        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)
        self.chunks = {c.chunk_id: c for c in chunks}
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        # Fingerprint the corpus; if the persisted collection matches, reuse
        # it instead of re-embedding (a 300-doc corpus takes ~20-40 min on
        # CPU — unacceptable on every app start).
        fp_input = "|".join(
            f"{c.chunk_id}:{hashlib.sha256(_searchable_text(c).encode()).hexdigest()}"
            for c in sorted(chunks, key=lambda item: item.chunk_id)
        )
        fp = hashlib.sha256(("metadata-v2|" + fp_input).encode()).hexdigest()[:16]
        try:
            col = client.get_collection("filings")
            if (col.metadata or {}).get("fingerprint") == fp and col.count() == len(chunks):
                self.col = col
                print(f"[index] reusing persisted index ({col.count()} chunks)")
                return
            client.delete_collection("filings")
        except Exception:
            pass

        self.col = client.create_collection(
            "filings", metadata={"hnsw:space": "cosine", "fingerprint": fp})
        texts = [c.text for c in chunks]
        embedding_texts = [_searchable_text(c) for c in chunks]
        embeddings = self.model.encode(embedding_texts, batch_size=32, show_progress_bar=True,
                                       normalize_embeddings=True)
        ids = [c.chunk_id for c in chunks]
        embs = embeddings.tolist()
        metas = [{"ticker": c.ticker, "form": c.form, "date": c.date,
                  "doc_id": c.doc_id, "item": c.item,
                  "section_title": c.section_title} for c in chunks]
        # Chroma caps a single .add() at ~5461 records — batch to stay under it
        # (unbatched, a 5,863-chunk corpus raises and silently falls back to TF-IDF).
        BATCH = 5000
        for i in range(0, len(ids), BATCH):
            self.col.add(
                ids=ids[i:i + BATCH],
                embeddings=embs[i:i + BATCH],
                documents=texts[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
            )

    def search(self, query: str, k: int, ticker: str | None = None, **filters):
        q = self.model.encode([query], normalize_embeddings=True).tolist()
        where = _chroma_where(ticker=ticker, **filters)
        n_results = max(k * 4, k)
        try:
            res = self.col.query(query_embeddings=q, n_results=n_results, where=where)
        except Exception:
            res = self.col.query(query_embeddings=q, n_results=n_results)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            # in-memory dict when we built this session; reconstruct from the
            # persisted store when only the Chroma directory is present
            chunk = self.chunks.get(cid) or self._chunk_from_store(cid)
            if not _matches_filters(chunk, ticker=ticker, **filters):
                continue
            out.append((chunk, 1.0 - dist))
            if len(out) >= k:
                break
        return out

    def _chunk_from_store(self, cid: str) -> Chunk:
        got = self.col.get(ids=[cid], include=["documents", "metadatas"])
        m = got["metadatas"][0]
        return Chunk(chunk_id=cid, text=got["documents"][0],
                     doc_id=m.get("doc_id", ""), ticker=m["ticker"],
                     form=m["form"], date=m["date"], item=m.get("item", ""),
                     section_title=m.get("section_title", ""))

    def filtered_chunks(self, ticker: str | None = None, **filters) -> list[Chunk]:
        return [c for c in self.chunks.values()
                if _matches_filters(c, ticker=ticker, **filters)]


class _ChromaIndex:
    """Manifest-driven Chroma loader.

    The legacy implementation above eagerly instantiated the embedding model
    and could rebuild the whole collection from inside the constructor. This
    definition intentionally shadows it so callers get explicit reuse/rebuild
    behavior without changing the public build_index API.
    """

    COLLECTION = "filings"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        rebuild_index: bool = False,
        require_existing: bool = False,
        require_query_encoder: bool = False,
        load_query_encoder: bool = True,
        doc_count: int | None = None,
    ):
        start = time.perf_counter()
        self.chunks = {c.chunk_id: c for c in chunks}
        self.model = None
        self.query_encoder_error = ""
        self.col = None
        self.info: IndexBuildInfo | None = None

        try:
            import chromadb
        except Exception as exc:
            raise IndexLoadError(f"Chroma dependency is unavailable: {exc}") from exc

        fp = corpus_fingerprint(chunks)
        manifest_file = index_manifest_path()
        manifest = _read_manifest(manifest_file)
        current = _current_manifest(chunks, doc_count=doc_count, fingerprint=fp)
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        collection = None
        collection_count: int | None = None
        collection_error = ""
        try:
            collection = client.get_collection(self.COLLECTION)
            collection_count = int(collection.count())
        except Exception as exc:
            collection_error = str(exc)

        if not rebuild_index:
            self.col = self._load_existing(
                collection=collection,
                collection_count=collection_count,
                collection_error=collection_error,
                manifest=manifest,
                current=current,
            )
            if load_query_encoder:
                self._load_query_encoder(require_query_encoder=require_query_encoder)
            self.info = self._build_info(
                status="existing index loaded",
                reused=True,
                regenerated=False,
                rebuild_requested=False,
                rebuild_performed=False,
                rebuild_reason="",
                manifest=manifest or current,
                fp=fp,
                doc_count=doc_count,
                start=start,
            )
            return

        self.col = self._rebuild(client, chunks, current)
        manifest = _read_manifest(manifest_file) or current
        self.info = self._build_info(
            status="rebuilt",
            reused=False,
            regenerated=True,
            rebuild_requested=True,
            rebuild_performed=True,
            rebuild_reason="explicit --rebuild-index",
            manifest=manifest,
            fp=fp,
            doc_count=doc_count,
            start=start,
        )

    def _load_existing(
        self,
        *,
        collection,
        collection_count: int | None,
        collection_error: str,
        manifest: dict[str, Any] | None,
        current: dict[str, Any],
    ):
        if collection is None:
            message = (
                "persistent Chroma collection is missing"
                + (f" ({collection_error})" if collection_error else "")
                + "; run with --rebuild-index to create it"
            )
            raise IndexLoadError(message)
        if collection_count != len(self.chunks):
            raise IndexLoadError(
                "persistent Chroma index is incomplete or stale: "
                f"collection has {collection_count} chunks, expected {len(self.chunks)}; "
                "run with --rebuild-index"
            )
        if manifest is None:
            raise IndexLoadError(
                f"index manifest missing at {index_manifest_path()}; run with --rebuild-index"
            )
        manifest_current = dict(current)
        manifest_current["embedding_dimension"] = manifest.get("embedding_dimension")
        reason = _manifest_mismatch(manifest, manifest_current)
        if reason:
            raise IndexLoadError(f"persistent Chroma index incompatible: {reason}; run with --rebuild-index")
        return collection

    def _load_query_encoder(self, *, require_query_encoder: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(settings.embedding_model)
        except Exception as exc:
            self.query_encoder_error = f"{type(exc).__name__}: {exc}"
            if require_query_encoder:
                raise IndexLoadError(
                    "persistent Chroma index loaded, but query embedding model "
                    f"{settings.embedding_model!r} is unavailable: {self.query_encoder_error}"
                ) from exc

    def _rebuild(self, client, chunks: list[Chunk], current: dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(settings.embedding_model)
            dimension_getter = getattr(
                self.model,
                "get_embedding_dimension",
                self.model.get_sentence_embedding_dimension,
            )
            dimension = dimension_getter()
        except Exception as exc:
            raise IndexLoadError(
                f"cannot rebuild index because embedding model {settings.embedding_model!r} "
                f"is unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        current = dict(current)
        current["embedding_dimension"] = dimension
        try:
            client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        collection = client.create_collection(
            self.COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "corpus_fingerprint": current["corpus_fingerprint"],
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "metadata_schema_version": INDEX_METADATA_SCHEMA_VERSION,
            },
        )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "ticker": c.ticker,
                "form": c.form,
                "date": c.date,
                "doc_id": c.doc_id,
                "item": c.item,
                "section_title": c.section_title,
            }
            for c in chunks
        ]
        embedding_texts = [_searchable_text(c) for c in chunks]
        batch = 512
        for i in range(0, len(ids), batch):
            embeddings = self.model.encode(
                embedding_texts[i:i + batch],
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            collection.add(
                ids=ids[i:i + batch],
                embeddings=embeddings.tolist(),
                documents=documents[i:i + batch],
                metadatas=metadatas[i:i + batch],
            )
            done = min(i + batch, len(ids))
            if done == len(ids) or done % (batch * 10) == 0:
                print(f"[index] embedded {done}/{len(ids)} chunks")

        _write_manifest(index_manifest_path(), current)
        return collection

    def _build_info(
        self,
        *,
        status: str,
        reused: bool,
        regenerated: bool,
        rebuild_requested: bool,
        rebuild_performed: bool,
        rebuild_reason: str,
        manifest: dict[str, Any],
        fp: str,
        doc_count: int | None,
        start: float,
    ) -> IndexBuildInfo:
        return IndexBuildInfo(
            backend="chroma",
            index_path=str(INDEX_DIR),
            index_status=status,
            index_reused=reused,
            embeddings_regenerated=regenerated,
            rebuild_requested=rebuild_requested,
            rebuild_performed=rebuild_performed,
            rebuild_reason=rebuild_reason,
            manifest_version=manifest.get("index_schema_version"),
            embedding_provider=manifest.get("embedding_provider", "sentence-transformers"),
            embedding_model=manifest.get("embedding_model", settings.embedding_model),
            embedding_dimension=manifest.get("embedding_dimension"),
            chunk_size=int(manifest.get("chunk_size", settings.chunk_size)),
            chunk_overlap=int(manifest.get("chunk_overlap", settings.chunk_overlap)),
            doc_count=doc_count if doc_count is not None else manifest.get("doc_count"),
            chunk_count=len(self.chunks),
            corpus_fingerprint=manifest.get("corpus_fingerprint", fp),
            load_time_seconds=round(time.perf_counter() - start, 3),
            query_encoder_loaded=self.model is not None,
            query_encoder_error=self.query_encoder_error,
        )

    def search(self, query: str, k: int, ticker: str | None = None, **filters):
        if self.model is None:
            return []
        q = self.model.encode([query], normalize_embeddings=True).tolist()
        where = _chroma_where(ticker=ticker, **filters)
        n_results = max(k * 4, k)
        try:
            res = self.col.query(query_embeddings=q, n_results=n_results, where=where)
        except Exception:
            res = self.col.query(query_embeddings=q, n_results=n_results)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            chunk = self.chunks.get(cid) or self._chunk_from_store(cid)
            if not _matches_filters(chunk, ticker=ticker, **filters):
                continue
            out.append((chunk, 1.0 - dist))
            if len(out) >= k:
                break
        return out

    def _chunk_from_store(self, cid: str) -> Chunk:
        got = self.col.get(ids=[cid], include=["documents", "metadatas"])
        m = got["metadatas"][0]
        return Chunk(
            chunk_id=cid,
            text=got["documents"][0],
            doc_id=m.get("doc_id", ""),
            ticker=m["ticker"],
            form=m["form"],
            date=m["date"],
            item=m.get("item", ""),
            section_title=m.get("section_title", ""),
        )

    def filtered_chunks(self, ticker: str | None = None, **filters) -> list[Chunk]:
        return [
            c for c in self.chunks.values()
            if _matches_filters(c, ticker=ticker, **filters)
        ]


class _TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True, max_features=50000)
        self.matrix = self.vec.fit_transform([_searchable_text(c) for c in chunks])

    def search(self, query: str, k: int, ticker: str | None = None, **filters):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if sims[i] <= 0:
                break
            if not _matches_filters(c, ticker=ticker, **filters):
                continue
            out.append((c, float(sims[i])))
            if len(out) >= k:
                break
        return out

    def filtered_chunks(self, ticker: str | None = None, **filters) -> list[Chunk]:
        return [c for c in self.chunks if _matches_filters(c, ticker=ticker, **filters)]


def _chroma_where(ticker: str | None = None, **filters):
    clauses = []
    if ticker:
        clauses.append({"ticker": {"$eq": ticker}})
    for field, value in filters.items():
        if field not in _FILTER_FIELDS:
            continue
        values = sorted(_as_filter_values(value))
        if not values:
            continue
        clauses.append({field: {"$eq": values[0]} if len(values) == 1 else {"$in": values}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _fallback_info(
    chunks: list[Chunk],
    *,
    error: str,
    doc_count: int | None = None,
    rebuild_index: bool = False,
) -> IndexBuildInfo:
    return IndexBuildInfo(
        backend="tfidf",
        index_path=str(INDEX_DIR),
        index_status="fallback",
        index_reused=False,
        embeddings_regenerated=False,
        rebuild_requested=rebuild_index,
        rebuild_performed=False,
        rebuild_reason=error,
        manifest_version=None,
        embedding_provider="sklearn",
        embedding_model="tfidf",
        embedding_dimension=None,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        doc_count=doc_count,
        chunk_count=len(chunks),
        corpus_fingerprint=corpus_fingerprint(chunks),
        load_time_seconds=0.0,
        query_encoder_loaded=False,
        query_encoder_error=error,
    )


def build_index(
    chunks: list[Chunk],
    *,
    rebuild_index: bool = False,
    require_persistent: bool = False,
    allow_tfidf_fallback: bool = True,
    doc_count: int | None = None,
    load_query_encoder: bool = True,
):
    """Return a hybrid (BM25 + dense) index.

    Normal evaluation passes require_persistent=True, which prevents silent
    corpus embedding regeneration or TF-IDF fallback. Callers that want to
    create corpus embeddings must pass rebuild_index=True explicitly.
    """
    lexical = _BM25(chunks)
    try:
        dense = _ChromaIndex(
            chunks,
            rebuild_index=rebuild_index,
            require_existing=require_persistent,
            require_query_encoder=require_persistent,
            load_query_encoder=load_query_encoder,
            doc_count=doc_count,
        )
        dense_name = "bge"
    except IndexLoadError as e:
        if require_persistent or not allow_tfidf_fallback:
            raise
        print(f"[index] semantic backend unavailable ({type(e).__name__}: {e}); "
              f"dense side falling back to TF-IDF")
        dense, dense_name = _TfidfIndex(chunks), "tfidf"
        dense.info = _fallback_info(
            chunks, error=f"{type(e).__name__}: {e}", doc_count=doc_count,
            rebuild_index=rebuild_index,
        )
    except Exception as e:  # no model cached / no chromadb / offline
        if require_persistent or not allow_tfidf_fallback:
            raise IndexLoadError(f"persistent semantic index failed to load: {e}") from e
        print(f"[index] semantic backend unavailable ({type(e).__name__}: {e}); "
              f"dense side falling back to TF-IDF")
        dense, dense_name = _TfidfIndex(chunks), "tfidf"
        dense.info = _fallback_info(
            chunks, error=f"{type(e).__name__}: {e}", doc_count=doc_count,
            rebuild_index=rebuild_index,
        )
    hybrid = HybridIndex(lexical, dense, dense_name)
    hybrid.index_info = getattr(dense, "info", None).as_dict()
    return hybrid, hybrid.name
