"""Phase 15 — scaling study. Where Phase 4's "approximation is free" claim goes to die.

Phase 4 benchmarked indexes on 25 chunks and found IVF/HNSW exactly matched exact Flat. That
is a true statement about 25 chunks and a useless one about production. This script rebuilds
the same indexes over a synthetic corpus at increasing scale (1k → 100k vectors) and measures
what actually changes: **build time, query latency, index memory, and recall vs the Flat
ground truth**.

Synthetic corpus: clustered random unit vectors (so ANN structure is meaningful — uniformly
random vectors are the pathological worst case for every ANN index and would understate them).
Ground truth = exact Flat top-k. Recall@k is measured *against Flat*, not qrels — at this
scale we're measuring the approximation, not the retrieval quality.

    python -m harness.scale_bench --sizes 1000 10000 50000 --dim 128 --k 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import modules  # noqa: F401  populate registry
from harness.contract import Chunk, Query
from harness.registry import build

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_corpus(n: int, dim: int, n_clusters: int = 50, seed: int = 0) -> list[Chunk]:
    """Clustered unit vectors — a realistic embedding geometry, not uniform noise."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.randint(0, n_clusters, size=n)
    vecs = centers[assign] + 0.35 * rng.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return [
        Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"chunk {i} cluster {assign[i]}",
              embedding=vecs[i])
        for i in range(n)
    ]


def make_queries(chunks: list[Chunk], n: int, dim: int, seed: int = 1) -> list[Query]:
    """Queries near real chunks (a query that matches nothing tells you nothing)."""
    rng = np.random.RandomState(seed)
    picks = rng.randint(0, len(chunks), size=n)
    qs = []
    for i, p in enumerate(picks):
        v = chunks[p].embedding + 0.15 * rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        q = Query(query_id=f"q{i}", text=chunks[p].text)
        q.embedding = v.astype(np.float32)
        qs.append(q)
    return qs


def index_memory_mb(name: str, n: int, dim: int, nlist: int = 0) -> float:
    """Analytic footprint of the stored vectors/graph (what dominates at scale)."""
    vec_mb = n * dim * 4 / 1e6                      # float32 vectors
    if name == "flat":
        return vec_mb
    if name == "ivf":
        return vec_mb + nlist * dim * 4 / 1e6       # + centroids
    if name == "hnsw":
        return vec_mb + n * 16 * 2 * 4 / 1e6        # + graph links (~M*2 neighbours, int32)
    return vec_mb


def run(sizes: list[int], dim: int, k: int, n_queries: int) -> list[dict]:
    rows = []
    for n in sizes:
        chunks = make_corpus(n, dim)
        queries = make_queries(chunks, n_queries, dim)

        # ── exact ground truth (Flat)
        flat = build("index", "flat")
        t = time.perf_counter(); flat.build(chunks); flat_build = (time.perf_counter() - t) * 1000
        lat, truth = [], []
        for q in queries:
            t = time.perf_counter(); res = flat.search(q, k); lat.append((time.perf_counter() - t) * 1000)
            truth.append({s.chunk.chunk_id for s in res})
        rows.append(dict(n=n, index="flat", build_ms=flat_build, p50_ms=float(np.percentile(lat, 50)),
                         p95_ms=float(np.percentile(lat, 95)), recall_vs_flat=1.0,
                         mem_mb=index_memory_mb("flat", n, dim)))

        # ── approximate indexes, scored against Flat
        nlist = max(8, int(np.sqrt(n)))
        for name, params in (("ivf", {"nlist": nlist, "nprobe": max(1, nlist // 16)}),
                             ("hnsw", {})):
            try:
                idx = build("index", name, **params)
            except KeyError:
                continue                     # hnswlib not installed → skip honestly
            t = time.perf_counter(); idx.build(chunks); b = (time.perf_counter() - t) * 1000
            lat, rec = [], []
            for q, gt in zip(queries, truth):
                t = time.perf_counter(); res = idx.search(q, k); lat.append((time.perf_counter() - t) * 1000)
                got = {s.chunk.chunk_id for s in res}
                rec.append(len(got & gt) / len(gt) if gt else 0.0)
            rows.append(dict(n=n, index=name, build_ms=b, p50_ms=float(np.percentile(lat, 50)),
                             p95_ms=float(np.percentile(lat, 95)),
                             recall_vs_flat=float(np.mean(rec)),
                             mem_mb=index_memory_mb(name, n, dim, nlist)))
        print(f"  … {n:,} vectors done", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-scale")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=50)
    a = ap.parse_args(argv)

    rows = run(a.sizes, a.dim, a.k, a.queries)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scaling_report.json").write_text(json.dumps(rows, indent=2))

    print(f"\n=== Scaling: {a.dim}-dim vectors, top-{a.k}, {a.queries} queries ===")
    print(f"{'n':>8} {'index':>6} {'build ms':>10} {'p50 ms':>9} {'p95 ms':>9} "
          f"{'recall vs flat':>15} {'mem MB':>8}")
    for r in rows:
        print(f"{r['n']:>8,} {r['index']:>6} {r['build_ms']:>10.1f} {r['p50_ms']:>9.3f} "
              f"{r['p95_ms']:>9.3f} {r['recall_vs_flat']:>15.3f} {r['mem_mb']:>8.1f}")
    print(f"\nReport → {RESULTS / 'scaling_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
