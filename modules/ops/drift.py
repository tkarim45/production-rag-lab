"""Drift detection (Phase 13) — from-scratch PSI over queries, embeddings and retrieval quality.

A RAG system degrades silently. The corpus is frozen but the *queries* move: new products,
new jargon, a marketing campaign that sends a different population at your box. Nothing
errors — recall just quietly falls. Drift detection is the smoke alarm.

**PSI (Population Stability Index)**, the industry-standard, model-free stability statistic:

    PSI = Σ_bins (actual% − expected%) · ln(actual% / expected%)

Standard thresholds (Siddiqi, credit-risk lineage — a *convention*, not a theorem):

    PSI < 0.10  → no meaningful shift
    0.10–0.25   → moderate shift, investigate
    PSI ≥ 0.25  → significant shift, act

It is symmetric-ish, non-negative, and zero iff the binned distributions are identical.
Implemented from scratch (numpy only) — the whole point is that it's ~15 lines and every
term is inspectable.

Three surfaces, one statistic:
- `query_drift`      — are we being asked different things? (query-embedding distribution)
- `embedding_drift`  — has the *representation* moved? (chunk/query vectors)
- `retrieval_quality_drift` — is recall@k falling on a rolling window? (the outcome metric)

Binning honesty: PSI's bins come from the **reference** sample. Quantile bins on a continuous
signal; when the reference has few distinct values (per-query recall is literally 0 or 1),
quantile binning collapses and PSI is computed **categorically** over the observed values
instead. Empty actual bins are floored at `epsilon` — without it PSI is +inf the moment one
bin empties, which on small samples is constantly. Both choices change the number; both are
declared here rather than buried.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Sequence

import numpy as np

# Siddiqi's conventional thresholds. Convention, not law — see the module README.
NO_SHIFT = 0.10
SIGNIFICANT_SHIFT = 0.25

# floor for empty bins: PSI is undefined (inf) when a bin has 0% on either side
EPSILON = 1e-6


def verdict(psi_value: float) -> str:
    """Map a PSI value onto the conventional 0.1 / 0.25 bands."""
    if psi_value < NO_SHIFT:
        return "none"
    if psi_value < SIGNIFICANT_SHIFT:
        return "moderate"
    return "significant"


def psi_from_proportions(
    expected_pct: Sequence[float], actual_pct: Sequence[float], epsilon: float = EPSILON
) -> float:
    """PSI from two aligned proportion vectors. The definition, verbatim, nothing else.

    PSI = Σ (a − e) · ln(a / e), with a, e floored at `epsilon` so an empty bin costs a
    large-but-finite penalty instead of +inf.
    """
    e = np.asarray(expected_pct, dtype=np.float64)
    a = np.asarray(actual_pct, dtype=np.float64)
    if e.shape != a.shape:
        raise ValueError(f"proportion vectors must align: {e.shape} vs {a.shape}")
    if e.size == 0:
        raise ValueError("need at least one bin")
    e = np.maximum(e, epsilon)
    a = np.maximum(a, epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(reference: Sequence[float], bins: int = 10) -> np.ndarray:
    """Bin edges from the reference sample's quantiles, open at both ends.

    Open ends (±inf) matter: an actual value outside the reference's range must land in an
    end bin, not be dropped — an out-of-range shift is exactly the drift you want to catch.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if ref.size == 0:
        raise ValueError("reference sample is empty")
    inner = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    edges = np.unique(np.concatenate(([-np.inf], inner, [np.inf])))
    return edges


def _proportions(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    counts = np.histogram(v, bins=edges)[0].astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def psi_categorical(expected: Iterable[Any], actual: Iterable[Any],
                    epsilon: float = EPSILON) -> float:
    """PSI over discrete values (the honest path for binary/low-cardinality signals)."""
    exp = list(expected)
    act = list(actual)
    if not exp or not act:
        raise ValueError("both samples must be non-empty")
    categories = sorted(set(exp) | set(act), key=str)
    e = np.array([exp.count(c) for c in categories], dtype=np.float64) / len(exp)
    a = np.array([act.count(c) for c in categories], dtype=np.float64) / len(act)
    return psi_from_proportions(e, a, epsilon=epsilon)


def psi(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
    edges: np.ndarray | None = None,
    epsilon: float = EPSILON,
) -> float:
    """PSI between a reference (`expected`) and a live (`actual`) 1-D sample.

    Binning: explicit `edges` if given; else quantile bins from `expected`; else — when the
    reference carries `<= bins` distinct values — categorical over the observed values,
    because quantile bins on {0, 1} produce one bin and a meaningless PSI of 0.
    """
    exp = np.asarray(expected, dtype=np.float64).ravel()
    act = np.asarray(actual, dtype=np.float64).ravel()
    if exp.size == 0 or act.size == 0:
        raise ValueError("both samples must be non-empty")

    if edges is None:
        distinct = np.unique(exp)
        if distinct.size <= bins:
            return psi_categorical(exp.tolist(), act.tolist(), epsilon=epsilon)
        edges = quantile_edges(exp, bins=bins)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        return 0.0
    return psi_from_proportions(_proportions(exp, edges), _proportions(act, edges), epsilon)


# ── the three drift surfaces ─────────────────────────────────────────────────


def _as_matrix(vectors) -> np.ndarray:
    m = np.asarray([np.asarray(v, dtype=np.float64).ravel() for v in vectors], dtype=np.float64)
    if m.ndim != 2 or m.size == 0:
        raise ValueError("need a non-empty 2-D stack of embeddings")
    return m


def embedding_drift(reference, actual, bins: int = 10) -> dict[str, Any]:
    """PSI on an embedding *distribution*. Two projections, because neither alone is enough.

    PSI is 1-D; embeddings are not. The two honest reductions:
    - `psi_centroid_cosine`: PSI of each vector's cosine similarity to the **reference
      centroid**. Cheap, interpretable ("the new queries point elsewhere"), but blind to a
      shift that is orthogonal to the centroid — it can read 0 while the cloud has moved.
    - `psi_per_dim_mean`: PSI computed per dimension and averaged. Sees axis-aligned shifts
      the projection misses, but dilutes a large shift in one dimension across `dim` of them.

    Report both; disagreement between them is information, not a bug.
    """
    ref = _as_matrix(reference)
    act = _as_matrix(actual)
    if ref.shape[1] != act.shape[1]:
        raise ValueError(f"embedding dims differ: {ref.shape[1]} vs {act.shape[1]}")

    centroid = ref.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        cos_ref = np.zeros(len(ref))
        cos_act = np.zeros(len(act))
    else:
        unit = centroid / norm
        def _cos(m):
            norms = np.linalg.norm(m, axis=1)
            norms[norms == 0] = 1.0
            return (m @ unit) / norms
        cos_ref, cos_act = _cos(ref), _cos(act)

    psi_cos = psi(cos_ref, cos_act, bins=bins)
    per_dim = [psi(ref[:, d], act[:, d], bins=bins) for d in range(ref.shape[1])]
    psi_dim = float(np.mean(per_dim))
    headline = max(psi_cos, psi_dim)
    return {
        "psi_centroid_cosine": psi_cos,
        "psi_per_dim_mean": psi_dim,
        "psi_per_dim_max": float(np.max(per_dim)),
        "psi": headline,
        "verdict": verdict(headline),
        "n_reference": int(len(ref)),
        "n_actual": int(len(act)),
        "dim": int(ref.shape[1]),
    }


def query_drift(reference_queries, actual_queries, embedder=None, bins: int = 10) -> dict[str, Any]:
    """Query-distribution drift: are we being asked a different kind of question?

    Accepts `Query` objects (embedded via `embedder` when they carry no vector) or raw
    embeddings. This is the leading indicator — it fires *before* answer quality drops.
    """
    def _vecs(queries):
        out = []
        for q in queries:
            if hasattr(q, "embedding"):
                if q.embedding is None:
                    if embedder is None:
                        raise ValueError("queries are unembedded and no embedder was passed")
                    q = embedder.encode_query(q)
                out.append(q.embedding)
            else:
                out.append(q)
        return out

    return embedding_drift(_vecs(reference_queries), _vecs(actual_queries), bins=bins)


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Rolling mean of a per-query metric — the shape ops actually alerts on."""
    v = list(values)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(v) < window:
        return []
    return [float(np.mean(v[i - window:i])) for i in range(window, len(v) + 1)]


def retrieval_quality_drift(
    reference_values: Sequence[float], actual_values: Sequence[float],
    window: int | None = None, bins: int = 10,
) -> dict[str, Any]:
    """Drift in the *outcome*: per-query recall@k before vs after.

    The lagging indicator, and the one with teeth — query drift only matters if quality
    follows. Per-query recall is usually near-binary, so PSI here is categorical over {0, 1}
    (or the observed recall values), i.e. it is fundamentally "did the hit rate move".
    That's a real limitation, stated rather than hidden: on a binary outcome PSI carries no
    more information than the mean delta, and the mean delta has a CI (see `ab.py`).
    """
    ref = list(reference_values)
    act = list(actual_values)
    value = psi(ref, act, bins=bins)
    out: dict[str, Any] = {
        "psi": value,
        "verdict": verdict(value),
        "reference_mean": float(np.mean(ref)),
        "actual_mean": float(np.mean(act)),
        "delta": float(np.mean(act) - np.mean(ref)),
        "n_reference": len(ref),
        "n_actual": len(act),
        "distinct_reference_values": int(np.unique(ref).size),
    }
    if window:
        out["rolling_reference"] = rolling_mean(ref, window)
        out["rolling_actual"] = rolling_mean(act, window)
    return out


# ── CLI: the Phase 13 drift demo ──────────────────────────────────────────────
#     python -m modules.ops.drift


def _demo_method_sanity(n: int, seed: int) -> None:
    """Does PSI + the 0.1/0.25 bands work at all? Verify at a sample size they were designed for."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.0, 1.0, n)
    same = rng.normal(0.0, 1.0, n)
    shifted = rng.normal(0.5, 1.0, n)
    wide = rng.normal(0.0, 2.0, n)
    print(f"\n=== 1. method sanity — synthetic gaussians, n={n:,} per sample ===")
    for label, actual in (("N(0,1) vs itself (identical array)", ref),
                          ("N(0,1) vs N(0,1)  [resampled]", same),
                          ("N(0,1) vs N(0.5,1) [mean shift]", shifted),
                          ("N(0,1) vs N(0,2)  [variance shift]", wide)):
        v = psi(ref, actual, bins=10)
        print(f"  {label:<38} PSI = {v:7.4f}  → {verdict(v)}")


def _demo_noise_floor(seed: int) -> None:
    """The number that decides whether PSI is usable here: PSI of a population against ITSELF.

    Split one sample from a single distribution in half. Any PSI above zero is pure sampling
    noise — there is no drift to find. That value is the floor every real signal must clear.
    """
    rng = np.random.default_rng(seed)
    print("\n=== 2. noise floor — split-half of ONE population (no drift exists) ===")
    print(f"  {'n per half':>10} " + " ".join(f"{f'bins={b}':>10}" for b in (5, 10)))
    for n in (10, 20, 50, 100, 500, 1000, 10_000):
        row = []
        for bins in (5, 10):
            # median over 25 splits — one split is itself noisy
            vals = [psi(rng.normal(0, 1, n), rng.normal(0, 1, n), bins=bins) for _ in range(25)]
            row.append(float(np.median(vals)))
        print(f"  {n:>10} " + " ".join(f"{v:>10.4f}" for v in row)
              + ("   <- this lab's n" if n == 20 else ""))
    print("  (rule of thumb: >= 10 observations per bin, i.e. n >= 100 for bins=10)")

    print("\n=== 3. epsilon sensitivity — the empty-bin floor is an arbitrary constant ===")
    a, b = rng.normal(0, 1, 10), rng.normal(0, 1, 10)
    print(f"  {'epsilon':>10} {'PSI (n=10, no real drift)':>28}   verdict")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        v = psi(a, b, bins=10, epsilon=eps)
        print(f"  {eps:>10.0e} {v:>28.4f}   {verdict(v)}")


def _demo_query_drift(seed: int) -> None:
    """The real thing: a frozen embedder, a moving query population."""
    from harness.data import load_dataset
    from harness.registry import build as build_component

    docs, docs_queries = load_dataset("builtin_docs")
    _, mini_queries = load_dataset("builtin_mini")

    # frozen production embedder: fit ONCE on the served corpus, never refit (as in prod)
    embedder = build_component("embedder", "tfidf")
    chunker = build_component("chunker", "fixed", size=60, overlap=10)
    embedder.encode_chunks(chunker.run(docs))

    ref_vecs = [embedder.encode_query(q).embedding for q in docs_queries]
    ood_vecs = [embedder.encode_query(q).embedding for q in mini_queries]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ref_vecs))
    half = len(ref_vecs) // 2
    a = [ref_vecs[i] for i in order[:half]]
    b = [ref_vecs[i] for i in order[half:]]

    print(f"\n=== 4. query-embedding drift — tfidf (dim={len(ref_vecs[0])}), builtin_docs ===")
    cases = [
        (f"identical set vs itself (n={len(ref_vecs)})", ref_vecs, ref_vecs),
        (f"split-half of SAME population (n={half} vs {len(ref_vecs) - half})", a, b),
        (f"builtin_docs vs builtin_mini queries (n={len(ref_vecs)} vs {len(ood_vecs)})",
         ref_vecs, ood_vecs),
    ]
    print(f"  {'case':<52} {'PSI(cos)':>9} {'PSI(dim)':>9} {'headline':>9}  verdict")
    for label, ref, act in cases:
        d = embedding_drift(ref, act)
        print(f"  {label:<52} {d['psi_centroid_cosine']:>9.4f} {d['psi_per_dim_mean']:>9.4f} "
              f"{d['psi']:>9.4f}  {d['verdict']}")


def _demo_retrieval_quality_drift() -> None:
    """The lagging indicator: per-query recall@5 under a healthy vs a degraded config."""
    from modules.ops import runs

    ref = runs.per_query(runs.run_results(runs.variant("embedder", "tfidf")), "recall@5")
    act = runs.per_query(runs.run_results(runs.variant("embedder", "hashing")), "recall@5")
    d = retrieval_quality_drift(ref, act, window=5)
    print("\n=== 5. retrieval-quality drift — per-query recall@5, tfidf → hashing ===")
    print(f"  reference mean {d['reference_mean']:.3f} → actual mean {d['actual_mean']:.3f}  "
          f"(delta {d['delta']:+.3f})")
    print(f"  distinct reference values: {d['distinct_reference_values']} "
          f"→ PSI computed categorically")
    print(f"  PSI = {d['psi']:.4f} → {verdict(d['psi'])}")
    print(f"  rolling recall@5 (window=5), reference: "
          f"{[round(x, 2) for x in d['rolling_reference']]}")
    print(f"  rolling recall@5 (window=5), actual:    "
          f"{[round(x, 2) for x in d['rolling_actual']]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-drift", description="PSI drift demo (Phase 13).")
    ap.add_argument("--n-synthetic", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _demo_method_sanity(args.n_synthetic, args.seed)
    _demo_noise_floor(args.seed)
    _demo_query_drift(args.seed)
    _demo_retrieval_quality_drift()
    print(f"\nbands: PSI < {NO_SHIFT} none | {NO_SHIFT}–{SIGNIFICANT_SHIFT} moderate | "
          f">= {SIGNIFICANT_SHIFT} significant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
