# Module: Baseline (Phase 0)

**Lesson.** Every benchmark needs a number to beat. This module is the **naive RAG** every later
phase is measured against: the simplest thing that works end-to-end, with no cleverness
anywhere. Its job is to be *beatable* — and to prove the harness runs before any technique is
added.

It is also the **key-free floor**: the whole pipeline runs offline on numpy + pyyaml, so tests
are deterministic and CI needs no credentials.

## Implemented
| stage | component | what it does |
|---|---|---|
| chunker | `fixed` | fixed-size word window + overlap |
| embedder | `hashing` | FNV-1a hashed bag-of-words → L2-normalized vector. Deterministic, no download |
| index | `flat` | brute-force cosine — **exact by construction**, so it's the recall=1.0 ground truth every ANN index (Phase 4/15) is scored against |
| retriever | `dense` | pass-through to the index |
| assembler | `concat` | numbered, citable `[n] (chunk_id)` blocks |
| generator | `extractive_mock` | picks the context sentence with max query overlap. No LLM |
| generator | `claude` | real Claude on AWS Bedrock, grounded + cited, reports real tokens/cost |

## Result — `builtin_docs` (13 docs / 20 queries)

| config | recall@1 | recall@5 | MRR | token_f1 | p50 latency | cost/query |
|---|--:|--:|--:|--:|--:|--:|
| naive (key-free) | 0.550 | 0.950 | 0.750 | 0.328 | 0.07 ms | $0 |
| naive_claude (Haiku 4.5) | 0.550 | 0.950 | 0.750 | 0.381 | 1344 ms | $0.0005 |
| *(for reference)* `capstone.yaml` | **0.950** | **1.000** | **1.000** | **0.423** | 1165 ms | $0.0007 |

## Honest findings
1. **The baseline is deliberately weak in exactly one place — the embedder** (`hashing`,
   recall@1 0.55). That turned out to be the most useful design decision in the repo: it gave
   Phases 7 and 9 a *weak first stage* to prove reranking (+40pts) and contextual retrieval
   (+10pts) against. A baseline that's already good teaches nothing.
2. **`extractive_mock` is a floor, not a substitute** — 0.232–0.328 token-F1 vs the real model's
   0.355–0.423. It's honest about what it is: a sentence-picker with no attention, which is why
   Phase 8 found lost-in-the-middle *unmeasurable* against it.
3. **Phase 0 already exposed the metric problem** that Phase 11 later proved: EM = 0.000 and
   token-F1 barely separated real Claude (0.381) from a one-line extractive baseline (0.328).
   The suspicion started here; the LLM judge (Phase 11) settled it.
4. **`flat` being exact is a feature, not laziness.** Every approximate index in Phases 4 and 15
   is scored against it — you cannot measure an approximation without ground truth.
