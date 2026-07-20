# 00: Overview

## Problem

RAG is easy to prototype and hard to productionize. Every layer, chunking, embedding,
indexing, retrieval, reranking, context assembly, generation, has a dozen viable
techniques, and the "best" one is dataset-, latency-, and cost-dependent. Tutorials teach
tricks in isolation; nobody benchmarks the whole stack on one harness. Teams ship a naive
RAG, watch it hallucinate, and can't tell whether the fix is better chunking, a reranker,
query decomposition, or a different index.

## What we build

A single repo that is simultaneously:

- **A course**, each production RAG technique explained with its tradeoff.
- **A benchmark suite**, every variant of every layer implemented and scored on one
  shared corpus + eval set + metric harness.
- **A production reference**, the security, observability, scaling, and governance layers
  that separate a demo from a system, each implemented and measured.
- **A composable pipeline**, a final configurable production RAG that stitches the winners
  together, with a leaderboard proving each choice.

## Goals

- Benchmark **every** technique in the curriculum map on a shared harness, same corpus,
  same eval set, same metrics, so comparisons are honest.
- Cover the full **production** surface, not just retrieval quality: latency, cost, memory,
  security, drift, governance, scaling.
- Prove **scaling to a million documents** with approximate/on-disk/streaming techniques.
- Keep it runnable on an **Apple M1 (8 GB)**; scale-out is an optional documented cloud
  step.
- Every phase ships **an honest results table**, including "this fancy method loses on
  this dataset."

## Non-goals

- Not a single-architecture demo, that's the existing `rag-architectures` repo (13
  architectures). This is the full *production lifecycle* around retrieval, at benchmark
  depth.
- Not tied to one vector-DB vendor, pluggable adapters; FAISS/hnswlib/DiskANN by default.
- Not a training project, embedding/reranker fine-tuning is a module, not the focus.

## Success criteria (definition of done)

| # | Criterion | Evidence |
|---|---|---|
| S1 | Shared harness runs any pipeline config and emits all metrics | one command → metrics JSON |
| S2 | Every chunking/retrieval/rerank/context/flow variant benchmarked | per-phase results table |
| S3 | Retrieval metrics (Recall@k, Precision@k, MRR, Hit-Rate, NDCG, MAP) computed correctly | unit-tested against known cases |
| S4 | End-to-end metrics (EM, F1, correctness, groundedness, hallucination rate) computed | eval report |
| S5 | Human-eval interface + inter-annotator agreement working | labeled batch + κ score |
| S6 | Security layer: injection-via-doc blocked, permission-aware retrieval enforced | red-team + access test |
| S7 | Ops: drift alert, A/B test, CI regression gate all functioning | dashboards + red PR |
| S8 | Million-doc scaling run completes (on-disk/approx), with latency/memory numbers | scaling report |
| S9 | Final composed pipeline + cross-layer leaderboard | the headline deliverable |

## Chosen corpus + eval set

Use a public benchmark with relevance judgments so retrieval metrics are grounded:

- **Retrieval**: a BEIR subset (e.g. FiQA, SciFact, NFCorpus), has qrels for
  Recall@k/NDCG. Plus a domain corpus (SEC filings / Wikipedia subset) for e2e.
- **End-to-end QA**: a dataset with gold answers (e.g. HotpotQA for multi-hop, Natural
  Questions / SQuAD subset for single-hop), enables EM/F1 + answer-correctness.
- **Synthetic**: generate an eval set from the corpus (question + gold chunk + gold answer)
  with contamination checks, for the domain corpus.

Commit the exact corpora + splits in Phase 0. Everything is measured against these.

## The honest questions this repo answers

- Does semantic chunking actually beat recursive chunking, or just cost more?
- Where does a reranker pay for its latency, and where doesn't it?
- Do query-transform methods (HyDE, expansion) help single-hop but fail multi-hop?
- What's the real recall/latency/memory Pareto across ANN indexes at 1M docs?
- How much does contextual retrieval reduce retrieval-failure rate on your corpus?
- What's the cost-per-correct-answer of each pipeline, not just the accuracy?
