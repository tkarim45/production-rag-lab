"""Retriever implementations (Phase 5).

`bind_corpus(chunks, index, embedder)` is called once at pipeline build so a retriever can
stand up its own sub-indexes (the BM25 side of hybrid/sparse). `retrieve(query, k)` is the
per-query call the pipeline routes through.
"""

from __future__ import annotations

import numpy as np

from harness.contract import Query, Scored
from harness.fusion import rrf_fuse, weighted_fuse
from harness.registry import build, register


@register("retriever", "sparse")
class SparseRetriever:
    """Pure BM25 lexical retrieval."""

    name = "sparse"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._bm25 = build("index", "bm25", k1=k1, b=b)

    def bind_corpus(self, chunks, index, embedder) -> None:
        self._bm25.build(chunks)

    def retrieve(self, query: Query, k: int) -> list[Scored]:
        return self._bm25.search(query, k)


@register("retriever", "hybrid")
class HybridRetriever:
    """Dense + BM25 fused. `fusion=rrf` (rank-based, default) or `weighted` (score, α)."""

    name = "hybrid"

    def __init__(self, fusion: str = "rrf", alpha: float = 0.5, overfetch: int = 3):
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        self.fusion, self.alpha, self.overfetch = fusion, alpha, overfetch
        self._bm25 = build("index", "bm25")
        self._dense = None

    def bind_corpus(self, chunks, index, embedder) -> None:
        self._dense = index          # the pipeline's dense index (already built)
        self._bm25.build(chunks)

    def retrieve(self, query: Query, k: int) -> list[Scored]:
        n = k * self.overfetch
        dense = self._dense.search(query, n)
        sparse = self._bm25.search(query, n)
        if self.fusion == "rrf":
            return rrf_fuse([dense, sparse], top=k)
        return weighted_fuse([dense, sparse], weights=[self.alpha, 1 - self.alpha], top=k)


@register("retriever", "mmr")
class MMRRetriever:
    """Maximal Marginal Relevance over a base retriever: pick the next doc maximizing
    λ·sim(d,q) − (1−λ)·max sim(d, already-selected). Trades some relevance for diversity."""

    name = "mmr"

    def __init__(self, base: str = "dense", lam: float = 0.6, overfetch: int = 4):
        self.lam, self.overfetch = lam, overfetch
        self._base_name = base
        self._base = None
        self._index = None

    def bind_corpus(self, chunks, index, embedder) -> None:
        self._index = index
        if self._base_name == "hybrid":
            self._base = build("retriever", "hybrid")
            self._base.bind_corpus(chunks, index, embedder)
        else:
            self._base = build("retriever", "dense")
            self._base.bind(index)

    def retrieve(self, query: Query, k: int) -> list[Scored]:
        pool = self._base.retrieve(query, k * self.overfetch)
        if not pool:
            return []
        q = query.embedding
        selected: list[Scored] = []
        remaining = list(pool)
        while remaining and len(selected) < k:
            best, best_score = None, -1e9
            for s in remaining:
                rel = float(np.dot(s.chunk.embedding, q)) if s.chunk.embedding is not None else s.score
                div = 0.0
                if selected:
                    div = max(
                        float(np.dot(s.chunk.embedding, t.chunk.embedding))
                        if s.chunk.embedding is not None and t.chunk.embedding is not None else 0.0
                        for t in selected
                    )
                mmr = self.lam * rel - (1 - self.lam) * div
                if mmr > best_score:
                    best, best_score = s, mmr
            selected.append(best)
            remaining.remove(best)
        return selected
