"""Retrievers. Phase 0 ships plain dense retrieval (delegate to the index). Phase 5 adds
sparse (BM25), hybrid, RRF/weighted fusion, MMR, and metadata filtering.

The dense retriever is a thin pass-through to the index so the pipeline's retrieve stage
has a stable seam; later retrievers (hybrid) hold references to multiple indexes and fuse.
"""

from __future__ import annotations

from harness.contract import Index, Query, Scored
from harness.registry import register


@register("retriever", "dense")
class DenseRetriever:
    name = "dense"

    def __init__(self, index: Index | None = None):
        # index is injected by the pipeline builder after the index stage is constructed
        self.index = index

    def bind(self, index: Index) -> None:
        self.index = index

    def retrieve(self, query: Query, k: int) -> list[Scored]:
        if self.index is None:
            raise RuntimeError("retriever has no index bound")
        return self.index.search(query, k)
