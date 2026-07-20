# Module: Chunking (Phase 2)

**Lesson.** Chunking sets the ceiling on retrieval: too big and one chunk mixes many topics
(diluting the embedding, hurting precision); too small and the answer is split across chunks
(hurting recall and losing context). The "best" chunker is dataset-dependent, the only way
to know is to benchmark them all on one corpus and read the numbers.

## Strategies implemented
`fixed` (Phase 0 baseline), `recursive`, `sentence`, `paragraph`, `structural` (heading-aware,
degrades to paragraph on plain text), `semantic` (embedding-similarity boundaries),
`parent_child` (small-to-big: index small, carry the parent paragraph for Phase 8).

## Result: `builtin_docs` (13 docs, 20 doc-level queries), hashing embed + flat index

`python -m harness.sweep --vary chunker --options fixed recursive sentence paragraph structural semantic parent_child`

| chunker | recall@1 | recall@5 | MRR | NDCG@10 | MAP | n_chunks |
|---|--:|--:|--:|--:|--:|--:|
| **recursive** | **0.850** | 0.950 | **0.917** | **0.925** | **0.917** | 26 |
| semantic | 0.650 | **1.000** | 0.825 | 0.870 | 0.825 | 45 |
| parent_child | 0.650 | **1.000** | 0.827 | 0.871 | 0.827 | 66 |
| sentence | 0.700 | 0.950 | 0.842 | 0.870 | 0.842 | 53 |
| paragraph | 0.700 | 0.900 | 0.813 | 0.835 | 0.813 | 13 |
| structural | 0.700 | 0.900 | 0.813 | 0.835 | 0.813 | 13 |
| fixed (naive) | 0.550 | 0.950 | 0.750 | 0.797 | 0.742 | 25 |

**Honest findings.**
1. **No single winner**, `recursive` wins ranking (MRR/NDCG/MAP, and recall@1 0.85: it puts
   the right doc *first* most often), but `semantic` and `parent_child` win **recall@5 (1.0)**:
   they never miss within 5, they just don't rank #1 as sharply. Which is "best" depends on
   whether the downstream generator sees top-1 or top-5.
2. **The naive `fixed` baseline is beaten by every structured chunker on recall@1**. Phase 0's
   default was the weakest ranker here (0.55). That's the point of Phase 2.
3. **More chunks ≠ better**, `parent_child` (66 chunks) and `paragraph` (13 chunks) reach the
   same recall@5 tier from opposite extremes; storage cost (chunk count) is a real axis the
   leaderboard tracks alongside quality.
4. **token_f1 is flat (~0.21 to 0.23) across all chunkers**, the extractive-mock generator can't
   convert better retrieval into a better answer, so answer-quality deltas are invisible until
   a real generator + LLM-judge (Phases 10 to 11). Retrieval metrics are where chunking shows up.

Caveat: 13-doc corpus, directions are real, exact ranks would shift on a larger set. The
`.[data]` BEIR swap (Phase 1 deferred item) is where these numbers get authoritative.
