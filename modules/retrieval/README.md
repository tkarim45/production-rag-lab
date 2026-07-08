# Module: Retrieval strategies (Phase 5)

**Lesson.** Given an index, *how* you query it is its own decision: dense (semantic), sparse
(lexical/BM25), hybrid (both, fused), and diversity control (MMR). Hybrid usually wins in
production — but only when the query mix actually splits the two retrievers.

## Implemented
- `dense` (Phase 0) — semantic, delegates to the vector index.
- `sparse` — pure BM25 lexical.
- `hybrid` — dense + BM25 fused by **RRF** (rank-based, no score normalization) or `weighted`
  (min-max-normalized convex score fusion, the fragile alternative shown for contrast).
- `mmr` — Maximal Marginal Relevance over a base retriever: `λ·sim(d,q) − (1−λ)·max sim(d,sel)`
  to control redundancy/diversity.

## Result — `builtin_docs`, tfidf embed + flat index

`python -m harness.sweep --vary retriever --options dense sparse hybrid hybrid_weighted mmr`

| retriever | recall@1 | recall@5 | p50 latency | note |
|---|--:|--:|--:|---|
| dense | 0.950 | 1.000 | 0.10 ms | saturates |
| sparse (BM25) | 0.950 | 1.000 | 0.13 ms | saturates |
| hybrid (RRF) | 0.950 | 1.000 | 0.16 ms | no gain, +60% latency |
| hybrid (weighted) | 0.950 | 1.000 | 0.17 ms | no gain |
| mmr (λ=0.6) | 0.950 | 0.975 | 0.82 ms | diversity costs recall + O(k²) |

**Honest findings.**
1. **Hybrid adds latency for zero recall gain *here*** — because dense already saturates
   (1.000) on a lexically clean corpus with no vocabulary mismatch. This is the honest,
   counter-hype result: **hybrid pays off only when the query mix splits the retrievers** —
   some queries need exact terms (BM25 wins) and others need paraphrase/semantics (dense wins).
   On a corpus where one retriever already gets everything, fusion is pure overhead. The
   senior answer to "should I use hybrid?" is "measure the split first," not "always."
2. **MMR trades recall for diversity and is the slowest** (O(k²) pairwise similarity). Use it
   only when redundant near-duplicate chunks are actually crowding your context — otherwise it
   demotes a relevant doc (recall@5 1.000 → 0.975) for diversity you didn't need.
3. **RRF vs weighted fusion:** identical quality here, but RRF needs no score normalization —
   the reason it's the production default (weighted fusion breaks when dense cosine and BM25
   scores live on different scales).

Caveat: to *show* hybrid winning you need a harder query set (vocabulary-mismatch + exact-term
queries). That's the `.[data]` BEIR swap — the mechanism is built and correct; the win is
corpus-dependent, exactly as the leaderboard says.
