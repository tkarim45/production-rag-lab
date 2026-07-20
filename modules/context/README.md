# Module: Context assembly (Phase 8)

**Lesson.** Retrieval hands you a ranked list; assembly decides what the model actually reads.
Which chunks, in what order, deduped how, under what token budget, and whether provenance
survives so the answer can cite. It's the most under-invested RAG stage.

## Implemented
- `concat` (Phase 0 baseline), numbered, citable blocks in rank order.
- `reorder`, **lost-in-the-middle** mitigation (Liu et al. 2023): models attend to the start
  and end of context and miss the middle, so put the best chunks at the *edges*.
- `dedup`, drop near-duplicate retrieved chunks (cosine ≥ threshold to a kept chunk).
- `budget`, pack best-first under a word budget (cost-aware; prevents silent truncation).
- `parent`, small-to-big: retrieve the precise child, feed the LLM its parent paragraph.

## Result: `builtin_docs`, parent_child chunker (66 chunks), extractive-mock generator

`python -m harness.sweep --vary assembler --chunker parent_child --options concat reorder dedup budget parent`

| assembler | recall@1 | recall@5 | MRR | token_f1 | p50 latency |
|---|--:|--:|--:|--:|--:|
| concat | 0.950 | 1.000 | 1.000 | 0.225 | 0.052 ms |
| reorder | 0.950 | 1.000 | 1.000 | 0.225 | 0.055 ms |
| dedup | 0.950 | 1.000 | 1.000 | 0.225 | 0.055 ms |
| budget | 0.950 | 1.000 | 1.000 | 0.225 | 0.054 ms |
| **parent** | 0.950 | 1.000 | 1.000 | **0.240** | 0.116 ms |

**Honest findings.**
1. **Assembly cannot move retrieval metrics, by construction.** Recall/MRR/NDCG are identical
   across all five because assembly runs *after* retrieval. Anyone reporting a recall lift from
   an assembly change has a bug. The only metric assembly can touch is answer quality.
2. **You cannot measure lost-in-the-middle with a mock generator.** `reorder` is identical to
   `concat` here, the extractive mock picks the max-overlap sentence and has no attention, so
   it has no U-shaped bias to mitigate. **The mitigation only matters for a model that has the
   bug.** This is a real methodological trap: Phase 8's headline lever is unmeasurable until a
   real LLM generator (Phase 10) with a long, distractor-heavy context. Honest status: the
   mechanism is implemented and unit-tested; the *effect size* is pending.
3. **`parent` is the only mover (+1.5 token-F1)**, because it's the one assembler that changes
   *what information is present*, not just its order. Small-to-big genuinely works: retrieve
   precise, generate with context.
4. `dedup` and `budget` were no-ops on this corpus (no near-dups; the budget never bound), a
   reminder that a technique's value is corpus-dependent, not universal.
