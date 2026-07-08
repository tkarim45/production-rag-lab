# Module: Indexing & vector stores (Phase 4)

**Lesson.** The index decides the recall / latency / memory / build-time tradeoff. Exact
(Flat) is the ground truth but O(N·d) per query; ANN indexes (IVF, HNSW, DiskANN) trade a
little recall for orders-of-magnitude speed at scale. Sparse (BM25) is a different axis
entirely — lexical, exact-term, no embeddings.

## Implemented (deps-light)
- `ivf` — from-scratch IVF: numpy k-means coarse quantizer partitions vectors into `nlist`
  cells; a query probes only the `nprobe` nearest cells. Approximate — misses when the answer
  sits in an unprobed cell.
- `bm25` — from-scratch Okapi BM25 inverted index (`k1`, `b`, smoothed IDF). Uses query text.
- `hnsw` — multi-layer NSW graph via hnswlib (registers only if installed).
- (`flat` exact lives in Phase 0 as the recall=1.0 reference.)

## Result — `builtin_docs`, tfidf embed

`python -m harness.sweep --vary index --options flat ivf ivf_aggressive bm25 hnsw`

| index | recall@1 | recall@5 | p50 latency | note |
|---|--:|--:|--:|---|
| flat (exact) | 0.950 | 1.000 | 0.100 ms | ground truth |
| ivf (nprobe 2) | 0.950 | 1.000 | 0.097 ms | approximation free here |
| bm25 (lexical) | 0.950 | 1.000 | 0.118 ms | ties dense on clean text |
| hnsw | 0.950 | 1.000 | 0.103 ms | approximation free here |
| **ivf_aggressive (nprobe 1)** | 0.950 | **0.975** | **0.051 ms** | **the tradeoff: −2.5pt recall, 2× faster** |

**Honest findings.**
1. **At this scale, approximation is free.** IVF (nprobe≥2) and HNSW match exact Flat — the
   whole reason ANN exists: it approximates well. The recall cost only appears when you push
   the approximation (nprobe=1 with more cells → 0.975 recall@5, but 2× faster). That single
   row *is* the ANN Pareto in miniature.
2. **BM25 ties dense on lexically clean text.** With TF-IDF-quality embeddings and no
   vocabulary mismatch, the sparse lexical index is as good as dense here — which sets up
   Phase 5's real question: *when does combining them actually help?*
3. **The recall/latency exchange rate is the deliverable**, not a single "best" index. On 25
   chunks none of this bites; the `.[data]` million-doc run (Phase 15) is where Flat becomes
   impossible and the IVF/HNSW/DiskANN choice earns its keep.
