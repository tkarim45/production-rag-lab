"""Indexing & vector stores (Phase 4).

Registers approximate + sparse indexes to benchmark against the exact Flat baseline
(Phase 0): `ivf` (from-scratch k-means coarse quantizer + nprobe), `bm25` (lexical inverted
index), and `hnsw` (via hnswlib, only if installed). Each reports build time + memory so the
recall / latency / memory / build Pareto is measurable, which is the whole point of Phase 4.
"""

from modules.indexing import indexes  # noqa: F401
