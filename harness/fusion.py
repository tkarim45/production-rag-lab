"""Rank-fusion utilities shared by hybrid retrieval (Phase 5) and multi-query (Phase 6).

Reciprocal Rank Fusion (Cormack et al., SIGIR 2009): score(d) = Σ_lists 1/(k + rank_i(d)),
rank 1-based. Rank-based so it needs no score normalization across heterogeneous lists
(dense cosine vs BM25 vs a second query) — the reason it's the production default.
"""

from __future__ import annotations

from harness.contract import Scored

RRF_K = 60  # Cormack's empirical default


def rrf_fuse(result_lists: list[list[Scored]], k: int = RRF_K, top: int | None = None) -> list[Scored]:
    """Fuse ranked lists of Scored by RRF. De-dupes by chunk_id, keeps a representative
    Scored whose .score is the fused RRF score. Highest fused score first."""
    fused: dict[str, float] = {}
    rep: dict[str, Scored] = {}
    for lst in result_lists:
        for rank, s in enumerate(lst, start=1):
            cid = s.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            rep.setdefault(cid, s)
    order = sorted(fused, key=lambda c: -fused[c])
    out = [Scored(chunk=rep[c].chunk, score=fused[c]) for c in order]
    return out[:top] if top else out


def weighted_fuse(
    result_lists: list[list[Scored]], weights: list[float], top: int | None = None
) -> list[Scored]:
    """Convex score fusion with min-max normalization per list (the fragile alternative to
    RRF — shown for contrast). weights sum need not be 1; normalized internally."""
    assert len(result_lists) == len(weights)
    agg: dict[str, float] = {}
    rep: dict[str, Scored] = {}
    for lst, w in zip(result_lists, weights):
        if not lst:
            continue
        scores = [s.score for s in lst]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for s in lst:
            norm = (s.score - lo) / span
            cid = s.chunk.chunk_id
            agg[cid] = agg.get(cid, 0.0) + w * norm
            rep.setdefault(cid, s)
    order = sorted(agg, key=lambda c: -agg[c])
    out = [Scored(chunk=rep[c].chunk, score=agg[c]) for c in order]
    return out[:top] if top else out
