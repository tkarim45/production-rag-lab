# Production RAG Lab — teach + benchmark every layer of production RAG

> A mega-capstone that **teaches** and **benchmarks** every component of a
> production-grade Retrieval-Augmented Generation system — from raw-document ingestion
> through chunking, embedding, indexing, retrieval, reranking, context assembly,
> generation, evaluation, serving, observability, security, and scaling to millions of
> documents. Every module is a lesson + a runnable benchmark + honest results on a shared
> harness. Runs on an Apple M1 (8 GB); scale-out steps documented as optional cloud bursts.

**Status:** 🚧 Phase 0 (harness) complete — the shared benchmark runs end-to-end. Phases
1–16 pending. The build order lives in [`TODO.md`](TODO.md) (master checklist) and
[`docs/02-roadmap.md`](docs/02-roadmap.md) (phased plan). Build **one phase at a time**.

```bash
make install                       # editable install (core = numpy + pyyaml only)
make test                          # 20 tests pass (metrics vs hand-computed + e2e smoke)
make bench configs/naive.yaml      # naive baseline, key-free → results/leaderboard.md
make bench-claude                  # same pipeline, real Claude Haiku on Bedrock (needs .env)
```

**Phase 0 baseline (built-in 12-doc/12-query labeled set):**

| config | recall@5 | mrr | ndcg@10 | token_f1 | em | p50 latency | cost/q |
|---|--:|--:|--:|--:|--:|--:|--:|
| naive (key-free extractive) | 1.000 | 1.000 | 1.000 | 0.328 | 0.000 | 0.07 ms | $0 |
| naive_claude (Haiku 4.5, Bedrock) | 1.000 | 1.000 | 1.000 | 0.381 | 0.000 | 1344 ms | $0.0005 |

*Honest finding already visible at Phase 0:* on this tiny lexical corpus, exact retrieval is
saturated (recall@5 = 1.0), and **EM/token-F1 barely separate a real grounded LLM from a
one-line extractive baseline** — Claude's verbose, cited answers don't string-match the
terse gold. That's the lexical-metric blind spot that motivates the calibrated LLM-judge in
Phase 11, and why the corpus needs to grow (Phase 1) before retrieval choices become
measurable. The harness exists to make exactly this kind of thing visible.

**Mega-capstone.** Bigger than the other three capstones combined in surface area.
Estimated effort: **9–15 months solo**, but every phase is independently valuable and
shippable on its own.

---

## The idea

Most RAG content teaches one trick (chunking, or reranking, or RAGAS) in isolation.
Production RAG is a *system of tradeoffs* — and the only way to know which choice wins is
to **benchmark them all on one corpus, one eval set, one harness**, and read the numbers.

This repo is that harness plus a full curriculum. Each topic ships:

1. **A lesson** (`README.md` in the module) — what it is, why it matters, the tradeoff.
2. **Runnable implementations** of every variant (e.g. 8 chunking strategies).
3. **A benchmark** scoring them on the shared metrics (retrieval quality, answer quality,
   latency, cost, memory).
4. **An honest results table** — the winner is dataset-dependent; the repo says so.

The final phase composes the best-of-every-layer into **one configurable production RAG
pipeline** with a leaderboard proving each choice.

## What's covered (nothing intentionally omitted)

Grouped; full taxonomy in [`docs/01-curriculum-map.md`](docs/01-curriculum-map.md).

- **Ingestion & parsing** — PDF/HTML/DOCX, layout-aware tables, OCR, cleaning,
  normalization, deduplication (exact + near-dup), metadata enrichment.
- **Chunking** — fixed, recursive, sentence, semantic, layout/structural, proposition,
  late chunking, parent-child / small-to-big, overlap tuning, chunk-size sweep.
- **Embeddings** — model selection (MTEB), dimensionality, Matryoshka truncation,
  int8/binary quantization, domain fine-tuning of embeddings, embedding caching.
- **Indexing / vector stores** — ANN indexes (Flat, IVF, IVFPQ, HNSW, DiskANN), the
  recall/latency/memory/build-time tradeoff, sparse (BM25) + learned-sparse (SPLADE).
- **Retrieval** — dense, sparse, **hybrid + fusion (RRF, weighted)**, metadata filtering,
  MMR / diversity, redundancy control.
- **Query understanding** — rewriting, expansion, HyDE, spelling correction,
  classification/routing, **decomposition, multi-hop, iterative/recursive retrieval**.
- **Reranking** — cross-encoder, ColBERT / late-interaction, LLM reranker, listwise,
  reciprocal-rank fusion of rerankers.
- **Context assembly** — selecting best chunks, **dedup of similar chunks, clustering**,
  ordering / lost-in-the-middle reordering, context compression, context-window budgeting.
- **Advanced flows** — GraphRAG, RAPTOR (hierarchical), **contextual retrieval**
  (Anthropic chunk-prefixing), Self-RAG, Corrective RAG (CRAG), Adaptive RAG, Agentic RAG.
- **Generation** — grounding & citation prompts, prompt engineering, **temperature /
  decoding params**, the LLM harness, abstention / no-answer / confidence.
- **Evaluation** — retrieval metrics (**Recall@k, Precision@k, MRR, Hit-Rate, NDCG, MAP**),
  end-to-end (**EM, F1, answer correctness, groundedness, hallucination rate**), RAGAS,
  LLM-as-judge (calibrated), **human-eval interface + inter-annotator agreement**,
  synthetic eval-set generation, contamination checks.
- **Serving** — API, async, streaming/TTFT, semantic + prompt + embedding caching,
  latency budgeting, **cost / token economics**.
- **Observability & ops** — tracing, monitoring, **embedding/query drift**, A/B testing
  configs, feedback loops (thumbs/click), **CI regression gate**.
- **Security & governance** — **prompt injection via retrieved docs / data poisoning**,
  PII redaction, **permission-aware / access-controlled retrieval**, multi-tenancy, audit
  logs, **right-to-be-forgotten (delete from index)**.
- **Scaling & maintenance** — **million-document** scale, sharding / distributed index,
  **incremental / real-time indexing (CDC)**, compaction, index versioning, storage
  tiering (memory ↔ disk / DiskANN), reindexing strategy.
- **Optional tracks** — multimodal RAG (images/tables), multilingual RAG, long-context vs
  RAG tradeoff, structured-data (SQL) + RAG hybrid.

### Production items added beyond the original brief

The original ask covered chunking/retrieval/indexing/ranking/reranking/metadata/eval/
multihop/iterative/decomposition/latency/storage/scaling. Added on top (commonly the
difference between a demo and production): **query understanding & routing, embedding
model selection + quantization + fine-tuning, ANN index tradeoffs, hybrid fusion,
contextual retrieval, self/corrective/adaptive/agentic RAG, semantic & prompt caching,
cost/token economics, drift monitoring, A/B testing, feedback loops, CI regression gates,
prompt-injection-via-documents defense, permission-aware retrieval, multi-tenancy, audit
logging, right-to-be-forgotten, incremental/real-time indexing, index versioning/
compaction, storage tiering, abstention/no-answer, multimodal & multilingual tracks.**

## Headline deliverable (what to demo)

A **cross-layer leaderboard**: for every decision point (chunker, embedder, index,
retriever, reranker, context strategy, RAG flow), a table + chart scoring each variant on
retrieval quality, answer quality, latency, cost, and memory on one shared corpus — plus a
final composed pipeline that stitches the winners together, with an honest "the winner is
dataset-dependent" writeup.

## What runs where (Apple M1, 8 GB)

| Concern | On the M1 | Optional cloud burst |
|---|---|---|
| Corpus | 10k–100k docs (representative) | million-doc scaling run |
| Embeddings | small sentence-transformers, int8/binary quantized | larger embedders |
| Vector index | FAISS Flat/IVF/HNSW, DiskANN for on-disk | distributed/sharded index |
| Generation | local Qwen2.5-1.5B (llama.cpp) + Claude API routed | — |
| Reranker/embedder fine-tune | MLX-LoRA (≤1.5B) | GPU fine-tune |

Scaling to a million docs is proven with **approximate + on-disk + streaming** techniques;
the laptop uses a smaller corpus but the *code path scales* — the big run is a documented
cloud step, not a laptop requirement.

## Repository layout (target)

```
production-rag-lab/
├── README.md
├── TODO.md                     # ← master phase checklist (start here)
├── docs/
│   ├── 00-overview.md          # problem, goals, success criteria
│   ├── 01-curriculum-map.md    # full taxonomy — every technique covered
│   ├── 02-roadmap.md           # phased build plan (Phase 0 → 16)
│   ├── 03-metrics-catalog.md   # every metric, defined + how computed
│   └── 04-setup.md             # M1 env, models, datasets
├── harness/                    # shared: dataset loaders, metrics, runner, leaderboard
├── modules/                    # one dir per topic (lesson + variants + benchmark)
├── configs/                    # composable pipeline configs
├── data/                       # gitignored; fetched by scripts
└── results/                    # leaderboards, charts, reports
```

## How to build it

1. Read [`docs/00-overview.md`](docs/00-overview.md) → [`docs/01-curriculum-map.md`](docs/01-curriculum-map.md).
2. Build **Phase 0 (the harness) first** — every later phase plugs into it. Nothing can be
   benchmarked without the shared harness + datasets + metrics.
3. Then work [`TODO.md`](TODO.md) / [`docs/02-roadmap.md`](docs/02-roadmap.md) **one phase
   at a time**. Each phase is a shippable mini-project with its own results.

## Tech stack

Python 3.12 · sentence-transformers · FAISS + hnswlib + DiskANN · rank-bm25 · SPLADE ·
cross-encoders / ColBERT · RAGAS · Claude (Anthropic/Bedrock) + local Qwen2.5-1.5B
(llama.cpp/MLX) · networkx (GraphRAG) · FastAPI · Redis · SQLite/DuckDB · Prometheus ·
MLflow/DVC · Streamlit/Next.js (human-eval UI + leaderboard) · Docker.

## License

Private. All rights reserved (personal portfolio project).
