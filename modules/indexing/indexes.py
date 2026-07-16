"""Index implementations (Phase 4). Exact Flat lives in modules/baseline.

- `ivf`  — IVF: k-means coarse quantizer partitions vectors into nlist cells; search probes
           the nprobe nearest cells only. Approximate — recall drops when the answer sits in
           an unprobed cell, which is exactly the tradeoff the leaderboard exposes.
- `bm25` — lexical inverted index with the Okapi BM25 score. Ignores embeddings; uses the
           query TEXT. The strong sparse baseline dense retrieval must beat.
- `hnsw` — multi-layer navigable-small-world graph via hnswlib (optional dep). Registers only
           if hnswlib is importable.

All implement the Index contract: build(chunks) / search(query, k).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from harness.contract import Chunk, Query, Scored
from harness.registry import register

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ── IVF (from-scratch) ────────────────────────────────────────────────────────


def _kmeans(x: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    """Tiny Lloyd's k-means. Init is deterministic (evenly-spaced picks, no RNG) so an index
    build is reproducible — an ANN benchmark whose recall moves between runs measures noise."""
    n = len(x)
    k = min(k, n)
    # k-means++-ish deterministic seed: evenly spaced picks
    idx = np.linspace(0, n - 1, k).astype(int)
    centroids = x[idx].copy()
    for _ in range(iters):
        # assign
        sims = x @ centroids.T
        assign = sims.argmax(axis=1)
        moved = False
        for c in range(k):
            members = x[assign == c]
            if len(members):
                new = members.mean(axis=0)
                nrm = np.linalg.norm(new)
                if nrm > 0:
                    new = new / nrm
                if not np.allclose(new, centroids[c]):
                    centroids[c] = new
                    moved = True
        if not moved:
            break
    return centroids


@register("index", "ivf")
class IVFIndex:
    """Inverted-file index: partition into nlist cells, probe nprobe at query time."""

    name = "ivf"

    def __init__(self, nlist: int = 8, nprobe: int = 2):
        self.nlist, self.nprobe = nlist, nprobe
        self._chunks: list[Chunk] = []
        self._centroids: np.ndarray | None = None
        self._cells: list[list[int]] = []

    def build(self, chunks: list[Chunk]) -> None:
        if any(c.embedding is None for c in chunks):
            raise ValueError("ivf needs embeddings")
        self._chunks = chunks
        mat = np.vstack([c.embedding for c in chunks]).astype(np.float32)
        self._mat = mat
        self._centroids = _kmeans(mat, self.nlist)
        assign = (mat @ self._centroids.T).argmax(axis=1)
        self._cells = [[] for _ in range(len(self._centroids))]
        for i, a in enumerate(assign):
            self._cells[a].append(i)

    def search(self, query: Query, k: int) -> list[Scored]:
        q = query.embedding
        cell_sims = self._centroids @ q
        probe = np.argsort(-cell_sims)[: self.nprobe]
        cand = [i for c in probe for i in self._cells[c]]
        if not cand:
            cand = list(range(len(self._chunks)))
        sims = self._mat[cand] @ q
        order = np.argsort(-sims)[:k]
        return [Scored(chunk=self._chunks[cand[i]], score=float(sims[i])) for i in order]


# ── BM25 (from-scratch Okapi) ─────────────────────────────────────────────────


@register("index", "bm25")
class BM25Index:
    """Okapi BM25 over an inverted index. score = Σ IDF(q)·(f·(k1+1))/(f + k1·(1−b+b·|d|/avgdl))."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._docs = [_tok(c.text) for c in chunks]
        self._len = np.array([len(d) for d in self._docs], dtype=np.float32)
        self._avgdl = float(self._len.mean()) if len(self._len) else 0.0
        self._tf = [Counter(d) for d in self._docs]
        df: Counter = Counter()
        for d in self._docs:
            for w in set(d):
                df[w] += 1
        n = len(chunks)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        # postings: term → list of doc ids
        self._postings: dict[str, list[int]] = {}
        for i, d in enumerate(self._docs):
            for w in set(d):
                self._postings.setdefault(w, []).append(i)

    def search(self, query: Query, k: int) -> list[Scored]:
        q = _tok(query.text)
        scores: dict[int, float] = {}
        for w in q:
            idf = self._idf.get(w)
            if idf is None:
                continue
            for i in self._postings.get(w, []):
                f = self._tf[i][w]
                denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avgdl or 1))
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / denom
        order = sorted(scores, key=lambda i: -scores[i])[:k]
        return [Scored(chunk=self._chunks[i], score=scores[i]) for i in order]


def _register_hnsw() -> None:
    try:
        import hnswlib  # noqa: F401
    except Exception:
        return

    @register("index", "hnsw")
    class HNSWIndex:  # pragma: no cover - needs optional dep
        name = "hnsw"

        def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
            self.M, self.efc, self.efs = M, ef_construction, ef_search

        def build(self, chunks):
            import hnswlib

            self._chunks = chunks
            mat = np.vstack([c.embedding for c in chunks]).astype(np.float32)
            self._p = hnswlib.Index(space="cosine", dim=mat.shape[1])
            self._p.init_index(max_elements=len(chunks), ef_construction=self.efc, M=self.M)
            self._p.add_items(mat, np.arange(len(chunks)))
            self._p.set_ef(self.efs)

        def search(self, query, k):
            labels, dist = self._p.knn_query(query.embedding, k=min(k, len(self._chunks)))
            return [Scored(chunk=self._chunks[int(i)], score=1.0 - float(d))
                    for i, d in zip(labels[0], dist[0])]


_register_hnsw()
