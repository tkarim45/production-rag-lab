# Module: Reranking (Phase 7)

**Lesson.** A reranker re-scores the first stage's candidates with a stronger, slower model.
It costs O(candidates) forward passes per query, so it must buy something. The rule:
**a reranker fixes *precision*, never *recall*.** If the answer isn't in the candidate set,
no reranker can save you; if it's in the set but ranked 7th, a reranker is the cheapest fix.

## Implemented
- `lexical`, deps-free pointwise BM25-style scorer over the candidate set. Same *shape* as a
  cross-encoder (re-score each pair independently, sort), no model download.
- `cross_encoder`, the real thing (`ms-marco-MiniLM`) via sentence-transformers (optional).
- `llm`, pointwise Claude/Bedrock relevance rating 0 to 10 (optional).

## Result: the whole point of Phase 7, in two tables

**A) On a WEAK first stage** (`hashing` embedder: recall@5 0.95 but recall@1 only 0.55, the
answer is retrieved but badly ranked. The reranker's ideal case):

`python -m harness.sweep --vary reranker --embedder hashing --options none lexical`

| reranker | recall@1 | recall@5 | MRR | NDCG@10 | p50 latency |
|---|--:|--:|--:|--:|--:|
| none | 0.550 | 0.950 | 0.750 | 0.797 | 0.097 ms |
| **lexical** | **0.950** | **1.000** | **1.000** | **1.000** | 0.251 ms |

**+40 points of recall@1** and a perfect MRR, for 2.6× latency.

**B) On a SATURATED first stage** (`tfidf`: already recall@1 0.95 / MRR 1.000):

`python -m harness.sweep --vary reranker --options none lexical`

| reranker | recall@1 | MRR | p50 latency |
|---|--:|--:|--:|
| none | 0.950 | 1.000 | 0.093 ms |
| lexical | 0.950 | 1.000 | 0.249 ms (**2.7×**) |

**Zero gain. Pure cost.**

**Honest findings.**
1. **The same reranker is a +40pt hero and a total waste, the difference is the first stage.**
   This is the single most transferable lesson in the repo, and it's measured on one harness,
   same corpus, same reranker, only the embedder swapped.
2. **"Should I add a reranker?" is not answerable without your recall/precision split.** High
   recall@k + low recall@1 → yes, big win. Already-high recall@1 → you're buying latency and
   nothing else. Measure before you add.
3. **Recall is the ceiling.** In (A) the reranker lifts recall@5 0.95→1.00 only because it
   reorders *within* the retrieved set, it can never retrieve what the first stage missed. If
   your recall@k is bad, fix retrieval (embedder/hybrid/chunking), not ranking.
