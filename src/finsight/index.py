from __future__ import annotations

import math
import re

from .chunker import Chunk
from .config import settings, INDEX_DIR

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
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
        import hashlib

        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)
        self.chunks = {c.chunk_id: c for c in chunks}
        client = chromadb.PersistentClient(path=str(INDEX_DIR))

        fp = hashlib.sha256("|".join(sorted(self.chunks)).encode()).hexdigest()[:16]
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
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True,
                                       normalize_embeddings=True)
        self.col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{"ticker": c.ticker, "form": c.form, "date": c.date} for c in chunks],
        )

    def search(self, query: str, k: int, ticker: str | None = None):
        q = self.model.encode([query], normalize_embeddings=True).tolist()
        where = {"ticker": ticker} if ticker else None
        res = self.col.query(query_embeddings=q, n_results=k, where=where)
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            out.append((self.chunks[cid], 1.0 - dist))
        return out


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
    lexical = _BM25(chunks)
    try:
        dense, dense_name = _ChromaIndex(chunks), "bge"
    except Exception as e:
        print(f"[index] semantic backend unavailable ({type(e).__name__}: {e}); falling back to TF-IDF")
        dense, dense_name = _TfidfIndex(chunks), "tfidf"
    hybrid = HybridIndex(lexical, dense, dense_name)
    return hybrid, hybrid.name
