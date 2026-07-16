"""Semantic cache (Phase 12) — a generator wrapper that serves a stored answer for a
sufficiently similar query.

An exact-match cache never fires in production: nobody types the same string twice. A
semantic cache fires when a new query's embedding is within `threshold` cosine of a cached
one, so "who invented the web" hits the entry for "Who invented the World Wide Web and when?"
The saving is the whole generation call — the most expensive stage in the pipeline.

**The number that matters is not the hit rate.** Every threshold you lower buys hits and buys
false hits: a *different* question, close in embedding space, served someone else's answer
with full confidence and zero cost. Hit rate alone always argues for a lower threshold; the
false-hit rate is what stops you. Both are reported (see `harness.cache_report`).

Two honest scope notes:
- Wrapping the **generator** caches the answer, not the retrieval — the pipeline still
  chunks, embeds, retrieves and assembles on a hit. That's the right seam here (generation is
  ~99% of the latency budget — see `modules/serving/README.md`), but a production cache in
  front of the whole pipeline would also skip retrieval.
- The entry is keyed on the *query* only. If the corpus changes, every cached answer is stale
  and there is no invalidation here — a real deployment versions the cache on the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from harness.contract import Query
from harness.registry import build, register


@dataclass
class _Entry:
    """One cached answer + the query vector it is served for."""

    vec: np.ndarray
    query_id: str
    text: str
    payload: dict[str, Any]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@register("generator", "semantic_cached")
class SemanticCachedGenerator:
    """Wrap any generator; serve a cached answer when cosine(query, cached) ≥ threshold.

    `embedder`: registry name of an embedder to key the cache with. Left `None`, the cache
    reuses the query embedding the pipeline already computed for retrieval (free, and it is
    the vector the system already trusts). If that is absent — e.g. a query handed straight to
    the generator — it degrades to a deterministic `hashing` embedder rather than failing.
    """

    name = "semantic_cached"

    def __init__(
        self,
        base: str = "extractive_mock",
        threshold: float = 0.95,
        embedder: str | None = None,
        **base_params,
    ):
        if not -1.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold is a cosine similarity; it must be in [-1, 1]")
        self.base = build("generator", base, **base_params)
        self.threshold = float(threshold)
        self._embedder = build("embedder", embedder) if embedder else None
        self._fallback = None
        self._entries: list[_Entry] = []
        self.hits = 0
        self.misses = 0

    # ── vectors ───────────────────────────────────────────────────────────────

    def _vec(self, query: Query) -> np.ndarray:
        if self._embedder is not None:
            # embed a copy: never clobber the retrieval embedding on the real query
            probe = self._embedder.encode_query(Query(query.query_id, query.text))
            return probe.embedding
        if query.embedding is not None:
            return query.embedding
        if self._fallback is None:
            self._fallback = build("embedder", "hashing")
        return self._fallback.encode_query(Query(query.query_id, query.text)).embedding

    def _nearest(self, vec: np.ndarray) -> tuple[int | None, float]:
        best_i, best_s = None, 0.0
        for i, e in enumerate(self._entries):
            s = _cosine(vec, e.vec)
            if best_i is None or s > best_s:
                best_i, best_s = i, s
        return best_i, best_s

    @property
    def stats(self) -> dict[str, Any]:
        looked_up = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / looked_up) if looked_up else 0.0,
            "entries": len(self._entries),
            "threshold": self.threshold,
        }

    # ── Generator interface ───────────────────────────────────────────────────

    def generate(self, query: Query, context: str) -> dict[str, Any]:
        vec = self._vec(query)
        idx, sim = self._nearest(vec)

        if idx is not None and sim >= self.threshold:
            e = self._entries[idx]
            self.hits += 1
            out = dict(e.payload)
            out["tokens"] = {"in": 0, "out": 0}     # a hit costs no tokens…
            out["cost_usd"] = 0.0                   # …and no money
            out["cache_hit"] = True
            out["cache_sim"] = sim
            out["cache_source_query_id"] = e.query_id
            out["cache_source_text"] = e.text
            return out

        self.misses += 1
        out = dict(self.base.generate(query, context))
        self._entries.append(_Entry(np.asarray(vec), query.query_id, query.text, dict(out)))
        out["cache_hit"] = False
        out["cache_sim"] = sim if idx is not None else 0.0
        out["cache_source_query_id"] = None
        out["cache_source_text"] = None
        return out
