"""Indexes. Phase 0 ships the exact Flat (brute-force cosine) index — the ground-truth
recall baseline every ANN index (Phase 4: IVF/IVFPQ/HNSW/DiskANN) is measured against."""

from __future__ import annotations

import numpy as np

from harness.contract import Chunk, Query, Scored
from harness.registry import register


@register("index", "flat")
class FlatIndex:
    """Brute-force cosine over a stacked matrix of L2-normalized embeddings.

    Exact by construction → this is the recall@k=1.0 reference. Slow at scale, which is the
    whole point of Phase 4's ANN Pareto study.
    """

    name = "flat"

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no embedding; run the embedder first")
        self._chunks = chunks
        self._matrix = np.vstack([c.embedding for c in chunks]).astype(np.float32)

    def search(self, query: Query, k: int) -> list[Scored]:
        if self._matrix is None:
            raise RuntimeError("index not built")
        if query.embedding is None:
            raise ValueError("query has no embedding")
        sims = self._matrix @ query.embedding  # cosine (both L2-normalized)
        k = min(k, len(self._chunks))
        # argpartition for top-k then sort those k
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Scored(chunk=self._chunks[i], score=float(sims[i])) for i in top]
