# Phase 15 — Scaling report

**Lesson.** Phase 4 benchmarked indexes on 25 chunks and found IVF/HNSW *exactly* matched
exact Flat. That is a true statement about 25 chunks and a useless one about production. This
phase re-runs the same indexes over synthetic clustered embeddings at 1k → 50k vectors and
measures what actually changes: **build time, query latency, memory, and recall against the
Flat ground truth**.

Method: clustered unit vectors (a realistic embedding geometry — uniformly random vectors are
the pathological worst case for every ANN index and would understate them), queries drawn near
real chunks, ground truth = exact Flat top-k, recall measured *vs Flat* (we're measuring the
approximation, not retrieval quality). `python -m harness.scale_bench`.

## Scaling curve (128-dim, top-10, 50 queries)

| n | index | build ms | p50 ms | p95 ms | recall vs flat | mem MB |
|--:|---|--:|--:|--:|--:|--:|
| 1,000 | flat | 0.6 | 0.027 | 0.038 | 1.000 | 0.5 |
| 1,000 | ivf | 10.9 | 0.018 | 0.023 | 0.134 | 0.5 |
| 1,000 | hnsw | 32.1 | 0.072 | 0.079 | 0.966 | 0.6 |
| 10,000 | flat | 4.7 | 0.174 | 0.304 | 1.000 | 5.1 |
| 10,000 | ivf | 180.7 | 0.137 | 0.190 | 0.324 | 5.2 |
| 10,000 | hnsw | 844.2 | 0.128 | 0.169 | 0.646 | 6.4 |
| 50,000 | flat | 25.0 | 0.951 | 1.094 | 1.000 | 25.6 |
| 50,000 | ivf | 1336.2 | 0.652 | 0.800 | 0.346 | 25.7 |
| 50,000 | hnsw | 7496.2 | 0.264 | 0.340 | 0.324 | 32.0 |

## The recall/latency dial at 50k (the real deliverable)

A single ANN row is meaningless without its tuning knob. Same corpus, sweeping `nprobe` (IVF)
and `ef_search` (HNSW):

| config | recall vs flat | p50 ms | speedup vs flat |
|---|--:|--:|--:|
| **flat (exact)** | **1.000** | 1.237 | 1.0× |
| ivf nlist=223 nprobe=13 | 0.346 | 0.912 | 1.4× |
| ivf nlist=223 nprobe=32 | 0.534 | 2.376 | **0.5× (slower)** |
| ivf nlist=223 nprobe=64 | 0.732 | 4.405 | **0.3× (slower)** |
| ivf nlist=223 nprobe=128 | 0.940 | 7.699 | **0.2× (slower)** |
| hnsw ef_search=50 | 0.334 | 0.260 | **4.8×** |
| hnsw ef_search=200 | 0.672 | 0.825 | 1.5× |
| hnsw ef_search=400 | 0.850 | 1.346 | 0.9× (slower) |

## Honest findings

1. **ANN is a *scale* technology, and below ~10k it is pure overhead.** At 1k vectors HNSW is
   **2.7× SLOWER** than exact Flat (0.072 vs 0.027 ms) — the graph traversal costs more than
   just scanning everything. The crossover only arrives around 10k–50k. Phase 4's "approximation
   is free" and this phase's "approximation is a tax" are the *same finding at different n*.

2. **Exact brute force is far more competitive than the ANN literature implies — because of
   BLAS.** Flat here is one `matrix @ vector` numpy call: 50k × 128 floats in **1.2 ms**. To beat
   that you need compiled SIMD code, not a better algorithm in Python.

3. **My from-scratch numpy IVF loses to exact Flat at every useful recall level** (0.53 recall @
   2.4 ms vs 1.00 recall @ 1.2 ms). It is *only* "faster" at nprobe=13, where recall is 0.346 —
   i.e. it's fast because it's wrong. **This is the honest headline: a pure-Python ANN is not an
   optimization, it's a regression.** FAISS and hnswlib exist because the constant factor *is*
   the product. Writing IVF from scratch taught the mechanism; it did not produce a usable index.

4. **hnswlib (real C++) is the only thing that beats Flat** — 4.8× at ef=50, 1.5× at ef=200. But
   at ef=400 (recall 0.85) it's *slower* than exact. At 50k on a laptop, **exact search is a
   legitimate production answer** and the ANN decision should be re-litigated, not assumed.

5. **Default `ef_search` does not scale.** HNSW recall collapses 0.966 (1k) → 0.646 (10k) →
   0.324 (50k) at a *fixed* ef_search=50. The parameter that was fine at 1k silently destroys
   recall at 50k. If you scale your corpus and don't re-tune ef/nprobe, your retrieval quality
   quietly rots — with no error, no alert, and a latency graph that looks great.

6. **Build time is the ANN tax nobody quotes.** HNSW build: 32 ms (1k) → 7,496 ms (50k) — a
   **234× increase for 50× the data**, superlinear. Flat "builds" in 25 ms. Every reindex
   (embedder change, Phase 15's blue-green) pays this.

7. **Memory is not where ANN hurts.** HNSW is only +25% over Flat (graph links); IVF +0.4%
   (centroids). At 50k both are ~26–32 MB. The vectors dominate — which is why *quantization*
   (Phase 3: int8 = 4× cut, lossless) is the memory lever, and ANN is the *latency* lever. Two
   different problems, two different tools.

## Caveats
Single M1 laptop, synthetic clustered vectors, 50 queries, n ≤ 50k. The million-doc /
DiskANN / sharding tier is the documented cloud burst (never a laptop requirement) — but the
code path is identical and the conclusion above ("re-tune ef/nprobe with n, and check whether
exact still wins") is what carries.
