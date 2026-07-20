# Phase 16: Capstone: the cross-layer master leaderboard

The headline deliverable: **every decision point, and the number that justifies it.** Built by
running each layer's variants on one harness, one corpus, one metric set (Phases 0 to 15), then
composing the winners into `configs/capstone.yaml`.

## The master leaderboard: every layer, its winner, and the measured delta

| Layer | Variants benchmarked | Winner | The number that justifies it |
|---|---|---|---|
| **Chunking** (P2) | fixed, recursive, sentence, paragraph, structural, semantic, parent_child | **recursive** | MRR **0.917** vs fixed 0.750; but semantic/parent_child win recall@5 (1.0), no universal winner |
| **Embedding** (P3) | hashing, tfidf, int8, binary, matryoshka 64/128/256 | **tfidf** (+int8 at scale) | recall@1 **0.55 → 0.95**. int8 = **lossless at 4× less memory**; binary **collapses to 0.05** |
| **Index** (P4/P15) | flat, ivf, ivf_aggressive, bm25, hnsw | **flat** (at this scale) | exact = **1.2 ms @ 50k** on BLAS. ANN only wins past ~10k *and* with compiled code |
| **Retrieval** (P5) | dense, sparse, hybrid(RRF), hybrid(weighted), mmr | **dense** | hybrid = **+60% latency, zero recall gain** here → *rejected* |
| **Query understanding** (P6) | none, prf, multiquery_prf | **none** | PRF **hurt**: recall@1 0.95 → 0.90 (query drift) → *rejected* |
| **Reranking** (P7) | none, lexical, cross-encoder, llm | **none** (on a strong stage) | zero gain at **2.7× latency** → *rejected*. On a **weak** stage: **+40pts recall@1** |
| **Context assembly** (P8) | concat, reorder, dedup, budget, parent | **concat** | reorder/dedup/budget were no-ops; parent +1.5 F1 (needs parent_child chunker) |
| **Advanced flows** (P9) | fixed vs contextual | **contextual** | weak-stage recall@1 **0.55 → 0.65**, MRR +6.7pts, ~free |
| **Generation** (P10/P11) | mock, bare, grounded, cite_forced, abstain, temp 0.7 | **cite_forced** | highest groundedness **0.902**, **100% citation rate**; judge-correctness 1.000 (tied) |
| **Evaluation** (P11) | EM, token-F1, LLM judge, groundedness, bootstrap CI | **judge + bootstrap** | EM is **statistically useless** (CI [0,0]); judge resolves it (+0.350, CI [+0.175,+0.550]) |

## The two capstone configs

| config | recall@1 | recall@5 | MRR | token_f1 | p50 latency | cost/query |
|---|--:|--:|--:|--:|--:|--:|
| Phase 0 naive (key-free) | 0.550¹ | 0.950 | 0.750 | 0.328 | 0.07 ms | $0 |
| Phase 0 naive_claude | 0.550¹ | 0.950 | 0.750 | 0.381 | 1344 ms | $0.0005 |
| **`capstone.yaml`** (best-of-every-layer + real Claude) | **0.950** | **1.000** | **1.000** | **0.423** | 1165 ms | $0.0007 |
| **`capstone_weak.yaml`** (weak embedder + contextual + rerank, offline) | **0.950** | **1.000** | **1.000** | 0.240 | **0.28 ms** | **$0** |

¹ the Phase 0 baseline used the `hashing` embedder.

**Capstone vs naive: recall@1 0.55 → 0.95 (+40pts), MRR 0.75 → 1.00, token-F1 0.381 → 0.423
(+11%), at −13% latency and +$0.0002/query.**

## The finding the whole repo was built to produce

> **Every technique only helps where there is headroom.**

Six of the nine levers were measured and **rejected**, hybrid retrieval, query expansion,
reranking, reordering, dedup, budgeting all did **nothing** on a saturated first stage. The
capstone is *mostly restraint*. That restraint is the deliverable: a config file where each
omission is backed by a number.

And the mirror image, `capstone_weak.yaml`, is the punchline:

> **On a weak first stage, contextual retrieval (+10pts) and a reranker (+40pts) recover
> recall@1 from 0.55 to 0.95, exactly matching the strong-embedder capstone, for $0 and
> 0.28 ms.** The same techniques the strong-stage capstone rejected are, on a weak stage, worth
> as much as replacing the embedder entirely.

So "should I use hybrid / rerank / expand / reorder?" has one correct answer: **measure your own
bottleneck first.** Both configs are in this repo; the only difference is where the headroom is.

## The harness caught its own bad claims: twice

The most valuable thing this repo produced is not a leaderboard row. It's that **the later
phases invalidated the earlier phases' headlines**, automatically, because they were built to:

| Claim | Made in | Overturned by | What actually held |
|---|---|---|---|
| "Prompt style is the best ROI in the lab, **+17% quality**" | Phase 10 (token-F1) | **Phase 11** (LLM judge) | Judge: all 4 styles = **1.000**, delta CI [0,0]. The +17% was a **lexical** delta, not correctness. What survives: prompt style moves **citation rate + groundedness**, free. |
| "Contextual retrieval: **+10pts recall@1**" | Phase 9 (point estimate) | **Phase 13** (paired bootstrap) | CI **[+0.000, +0.250]**, touches zero. Direction right (matches Anthropic), **claim not licensed at n=20**. |

Both corrections are annotated *in place* in the original phase READMEs rather than quietly
edited away. A harness that only confirms you is a vanity metric; this one argues back.

**The n=20 resolution limit is the meta-caveat for every number above.** Phase 13 measured it:
CI half-width ±0.10 to 0.20, so **nothing under ~20 points is resolvable on this eval set.** Read
every delta in the master leaderboard through that lens, the ones that survive it are the
embedder swap (+40pts), the reranker on a weak stage (+40pts), and the judge-vs-mock gap
(+0.350). The rest are directional.

## Honest caveats (the whole report in one paragraph)

13-doc corpus, 20 queries, single M1. Every "rejected" verdict above is **corpus-conditional,
not universal**, they were rejected *because tfidf saturates this corpus*, which is a property
of a small clean lexical set, not of production RAG. The `.[data]` BEIR/HotpotQA swap is the
single highest-value next step: it would restore headroom and very likely flip several verdicts
(hybrid and multi-hop especially). Phase 11's judge saturated at 1.000 for the same reason and
**overturned Phase 10's own headline**, proof the harness catches its own bad claims. Nothing
here is a benchmark result about RAG; it is a demonstration of the *method* for getting one.

## Deferred, stated plainly
GraphRAG / RAPTOR / Self-RAG / CRAG / Adaptive / Agentic (need an LLM control loop **and** a
corpus where retrieval fails often enough to route around, see `rag-architectures`);
IVFPQ/DiskANN/SPLADE/ColBERT; million-doc cloud burst; human-labeled judge calibration (κ);
context compression. Each is deferred with a reason, not silently dropped.
