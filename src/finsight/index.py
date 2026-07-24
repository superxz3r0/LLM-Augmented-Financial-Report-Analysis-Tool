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

import math
import re

from .chunker import Chunk
from .config import settings, INDEX_DIR

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
    """Okapi BM25 (k1=1.5, b=0.75). ~50 lines, fully explainable in an
    interview — that is the point of implementing it rather than importing."""

    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
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

    def search(self, query: str, k: int, ticker: str | None = None):
        q = _tokenize(query)
        scores = []
        for i, counts in enumerate(self.tf):
            if ticker and self.chunks[i].ticker != ticker:
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


class HybridIndex:
    """Reciprocal Rank Fusion over lexical and dense rankings:
    rrf(d) = sum over rankers of 1 / (60 + rank_d). Rank-based, so the two
    incomparable score scales never need calibrating."""

    RRF_K = 60

    def __init__(self, lexical, dense, dense_name: str):
        self.lexical, self.dense = lexical, dense
        self.name = f"hybrid(bm25+{dense_name})"

    def search(self, query: str, k: int, ticker: str | None = None):
        pool = max(k * 2, 10)
        ranked_lists = [
            self.lexical.search(query, pool, ticker=ticker),
            self.dense.search(query, pool, ticker=ticker),
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


class _ChromaIndex:
    def __init__(self, chunks: list[Chunk]):
        import chromadb

        from .config import get_embedder
        self.model = get_embedder()
        self.chunks = {c.chunk_id: c for c in chunks}
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        # Incremental indexing: diff this corpus's chunk ids against what's
        # already persisted and only embed what's new / delete what's gone,
        # instead of fingerprinting the whole corpus as one blob and
        # re-embedding everything the moment a single chunk changes. Under
        # the old whole-corpus-fingerprint scheme, adding one more company's
        # filings meant re-embedding every already-indexed company's filings
        # too — a 300-doc corpus took 20-40 min on CPU for what should have
        # been a handful of new documents. Chunk ids are stable across runs
        # (doc_id + a positional index — see chunker.py), so a plain id diff
        # is enough; this doesn't catch a filing's content changing in place
        # under an unchanged id, which the old fingerprint-by-id scheme
        # didn't catch either.
        self.col = client.get_or_create_collection("filings", metadata={"hnsw:space": "cosine"})
        existing_ids = set(self.col.get(include=[])["ids"])
        wanted_ids = set(self.chunks)
        to_remove = existing_ids - wanted_ids
        to_add = wanted_ids - existing_ids

        # Chroma caps a single call at ~5461 records — batch to stay under it
        # (unbatched, a 5,863-chunk corpus raises and silently falls back to TF-IDF).
        BATCH = 5000
        if to_remove:
            to_remove = list(to_remove)
            for i in range(0, len(to_remove), BATCH):
                self.col.delete(ids=to_remove[i:i + BATCH])

        if not to_add:
            print(f"[index] reusing persisted index ({len(wanted_ids)} chunks, "
                  f"{len(to_remove)} removed)")
            return

        print(f"[index] embedding {len(to_add)} new chunks "
              f"({len(wanted_ids) - len(to_add)} already indexed, {len(to_remove)} removed)")
        new_chunks = [self.chunks[cid] for cid in to_add]
        texts = [c.text for c in new_chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True,
                                       normalize_embeddings=True)
        ids = [c.chunk_id for c in new_chunks]
        embs = embeddings.tolist()
        metas = [{"ticker": c.ticker, "form": c.form, "date": c.date,
                  "doc_id": c.doc_id, "item": c.item,
                  "section_title": c.section_title} for c in new_chunks]
        for i in range(0, len(ids), BATCH):
            self.col.add(
                ids=ids[i:i + BATCH],
                embeddings=embs[i:i + BATCH],
                documents=texts[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
            )

    def search(self, query: str, k: int, ticker: str | None = None):
        q = self.model.encode([query], normalize_embeddings=True).tolist()
        where = {"ticker": ticker} if ticker else None
        res = self.col.query(query_embeddings=q, n_results=k, where=where)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            # in-memory dict when we built this session; reconstruct from the
            # persisted store when only the Chroma directory is present
            chunk = self.chunks.get(cid) or self._chunk_from_store(cid)
            out.append((chunk, 1.0 - dist))
        return out

    def _chunk_from_store(self, cid: str) -> Chunk:
        got = self.col.get(ids=[cid], include=["documents", "metadatas"])
        m = got["metadatas"][0]
        return Chunk(chunk_id=cid, text=got["documents"][0],
                     doc_id=m.get("doc_id", ""), ticker=m["ticker"],
                     form=m["form"], date=m["date"], item=m.get("item", ""),
                     section_title=m.get("section_title", ""))


class _TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True, max_features=50000)
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int, ticker: str | None = None):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if ticker and c.ticker != ticker:
                continue
            out.append((c, float(sims[i])))
            if len(out) >= k:
                break
        return out


def build_index(chunks: list[Chunk]):
    """Return a hybrid (BM25 + dense) index; dense backend is the best
    available: semantic embeddings, else TF-IDF."""
    lexical = _BM25(chunks)
    try:
        dense, dense_name = _ChromaIndex(chunks), "bge"
    except Exception as e:  # no model cached / no chromadb / offline
        print(f"[index] semantic backend unavailable ({type(e).__name__}: {e}); "
              f"dense side falling back to TF-IDF")
        dense, dense_name = _TfidfIndex(chunks), "tfidf"
    hybrid = HybridIndex(lexical, dense, dense_name)
    return hybrid, hybrid.name