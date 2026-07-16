# Production RAG Lab — teach + benchmark every layer of production RAG

> A mega-capstone that **teaches** and **benchmarks** every component of a
> production-grade Retrieval-Augmented Generation system — from raw-document ingestion
> through chunking, embedding, indexing, retrieval, reranking, context assembly,
> generation, evaluation, serving, observability, security, and scaling to millions of
> documents. Every module is a lesson + a runnable benchmark + honest results on a shared
> harness. Runs on an Apple M1 (8 GB); scale-out steps documented as optional cloud bursts.

**Status:** ✅ **All 17 phases complete** — harness, ingestion, chunking, embeddings,
indexing/ANN, retrieval, query understanding, reranking, context assembly, contextual retrieval,
generation, evaluation (LLM judge), serving, ops, security, scaling, capstone. Every phase ships
a lesson, a real benchmark, and an honest results table. LLM phases run on **real Claude Haiku
(AWS Bedrock)**. Headline deliverable: [`docs/06-capstone-report.md`](docs/06-capstone-report.md).
Optional tracks (multimodal / multilingual / long-context / SQL+RAG) remain open — see
[`TODO.md`](TODO.md).

```bash
make install                       # editable install (core = numpy + pyyaml only, no downloads)
make test                          # 178 tests pass (metrics vs hand-computed + e2e + per-phase)
make bench CONFIG=configs/capstone.yaml   # the composed best-of-every-layer pipeline
make bench configs/naive.yaml      # naive baseline, key-free → results/leaderboard.md
make bench-claude                  # same pipeline, real Claude Haiku on Bedrock (needs .env)
python -m harness.ingest data/raw_samples                       # Phase 1 ingestion report
python -m harness.sweep --vary chunker --options fixed recursive sentence paragraph structural semantic parent_child   # Phase 2
python -m harness.sweep --vary embedder --options hashing tfidf quantized_int8 quantized_binary matryoshka_64 matryoshka_128 matryoshka_256   # Phase 3
```

### Phases done — headline honest findings

- **Phase 0 (harness + naive baseline).** Config-driven swappable pipeline
  (chunk→embed→index→retrieve→rerank→assemble→generate), full metrics module (retrieval /
  answer / efficiency) unit-tested vs hand-computed cases, runner + leaderboard, naive baseline
  (key-free **and** real Claude Haiku on Bedrock). *Finding:* on the toy set, EM/token-F1 barely
  separate real Claude from a one-line extractive baseline — the lexical-metric blind spot that
  motivates the Phase 11 LLM-judge.
- **Phase 1 (ingestion).** Deps-free parsers (txt/md/html/csv; pdf/docx optional), cleaning,
  exact + near-dup (from-scratch MinHash) dedup, quality report. *Finding:* normalization policy
  decides "exact" vs "near" — a reworded near-dup folded to an exact dup after punctuation
  normalization.
- **Phase 2 (chunking).** 7 chunkers benchmarked on a 13-doc corpus with doc-level qrels.
  *Finding:* **no single winner** — `recursive` wins ranking (MRR 0.917), `semantic`/`parent_child`
  win recall@5 (1.0); the naive `fixed` chunker is the weakest ranker (recall@1 0.55).
- **Phase 3 (embeddings).** `tfidf` + int8/binary quantization + Matryoshka truncation.
  *Finding:* embedder choice dominates (recall@1 0.55→0.95 vs hashing); **int8 quant is lossless
  at 4× less memory**, but **binary quant collapses** (0.05) on sparse lexical vectors — the
  "32× smaller!" headline hides a broken index.
- **Phase 4 (indexing/ANN).** from-scratch IVF + BM25 + HNSW (optional) vs exact Flat.
  *Finding:* approximation is free at this scale (IVF/HNSW = Flat); push it (nprobe=1) and you
  buy 2× speed for −2.5pt recall@5 — the ANN Pareto in one row. BM25 ties dense on clean text.
- **Phase 5 (retrieval).** dense / sparse / hybrid (RRF + weighted) / MMR. *Finding:* hybrid
  adds latency for **zero** recall gain when dense already saturates — it pays off only when the
  query mix splits the retrievers. RRF matches weighted fusion but needs no score normalization.
- **Phase 6 (query understanding).** PRF expansion + multi-query (+ HyDE via Bedrock).
  *Finding:* PRF **hurts** a well-served corpus (recall@1 0.95→0.90) via query-drift — expansion
  fixes *under*-retrieval, not good retrieval. Measure the query type before adding it.
- **Phase 7 (reranking).** lexical/cross-encoder/LLM rerankers. ⭐ *The cleanest finding here:*
  on a **weak** first stage rerank lifts recall@1 **0.55 → 0.95 (+40pts)** for 2.6× latency; on
  a **saturated** stage the same reranker gives **zero gain at 2.7× latency**. Same harness,
  same reranker — only the first stage differs. **Rerankers fix precision, never recall.**
- **Phase 8 (context assembly).** reorder / dedup / budget / parent. *Finding:* assembly
  **cannot** move retrieval metrics (it runs after retrieval — a lift there is a bug); `parent`
  expansion is the only mover (+1.5 F1). **Lost-in-the-middle is unmeasurable with a mock
  generator** — the mitigation only matters for a model that has the bug.
- **Phase 9 (contextual retrieval).** Anthropic's chunk-prefixing, deps-free + LLM variants.
  *Finding:* prefixing the title lifts weak-stage recall@1 **0.55 → 0.65** at ~zero cost — the
  *free structural* context captured the value without an LLM call.
- **Phase 10 (generation, real Claude Haiku).** 4 prompt styles × temperature.
  *Finding:* **prompt style is the best ROI in the lab** — bare→abstain = **+6.2 token-F1 (+17%)
  at identical cost**. Forcing citations *improves quality* (0.355→0.409), not just auditability;
  citation rate 0%→30% (spontaneous)→100%. Temp 0 vs 0.7 is a wash on quality — temp 0 buys
  *determinism*, not accuracy. **EM = 0.000 for every real config** — EM is misleading for RAG.

- **Phase 11 (evaluation, real LLM judge).** G-Eval CoT judge + groundedness + context P/R +
  paired bootstrap CI. *Finding:* **EM is statistically useless** (0.000 everywhere, CI [0,0]);
  the judge resolves it (bare−mock **+0.350**, CI [+0.175,+0.550]). **Groundedness inverts the
  ranking** — the extractive mock is 1.000 grounded (it copies) but only 0.650 correct.
- **Phase 12 (serving).** Caches + latency budgeting. *Finding:* **generation is 99.99% of the
  p50 latency budget** (1287 of 1287.2 ms); retrieval is 0.006%. **Every Phase 4–7 retrieval
  optimization is invisible end-to-end** — they buy quality, not latency. Only a mock generator
  makes retrieval look worth optimizing.
- **Phase 13 (ops).** Tracing, PSI drift, A/B with bootstrap CI, CI gate. *Finding:* **PSI at
  n=20 measures sample size, not drift** — the split-half noise floor (where drift is impossible
  by construction) is **2.65**, 10× the 0.25 "alert" threshold.
- **Phase 14 (security).** Injection screen, PII+Luhn, ACL retrieval, RTBF. *Finding:* **every
  "filter it on the way out" design measured here leaks.** Post-hoc ACL filtering scrubs the
  citation list while the answer still repeats the secret verbatim — *it passes inspection*.
- **Phase 15 (scaling).** 1k→50k vectors + the recall/latency dial. *Finding:* **ANN is a scale
  technology** — at 1k, HNSW is **2.7× SLOWER** than exact Flat. And the **from-scratch numpy IVF
  loses to exact Flat at every useful recall level** (Flat is one BLAS matmul): a pure-Python ANN
  isn't an optimization, it's a regression.
- **Phase 16 (capstone).** Composed best-of-every-layer: recall@1 **0.55 → 0.95**, MRR 0.75 →
  1.00, token-F1 0.381 → **0.423**. **Six of nine levers were measured and REJECTED** — the
  capstone is mostly *restraint*, each omission backed by a number.

## ⭐ The two things this repo actually proves

**1. Every technique only helps where there is *headroom*.** Hybrid, PRF, reranking, reordering,
and contextual retrieval all did *nothing* on a saturated first stage and *a lot* on a weak one.
The punchline is [`configs/capstone_weak.yaml`](configs/capstone_weak.yaml): on a weak stage,
contextual (+10pts) + a reranker (+40pts) recover recall@1 **0.55 → 0.95 — matching the
strong-embedder capstone — for $0 and 0.28 ms.** "Should I add X?" is unanswerable without
measuring your own bottleneck.

**2. The harness caught its own bad claims — twice.**

| Claim | Made in | Overturned by | What held |
|---|---|---|---|
| "Prompt style = **+17% quality**" | Phase 10 (token-F1) | **Phase 11** (judge) | Judge: all styles **1.000**, CI [0,0]. It was a **lexical** delta, not correctness. |
| "Contextual retrieval **+10pts**" | Phase 9 | **Phase 13** (bootstrap) | CI **[+0.000, +0.250]** — touches zero. Directional, **not licensed at n=20**. |

Both are annotated *in place* in the original phase READMEs, not quietly edited away. A harness
that only confirms you is a vanity metric. This one argues back — which is the whole point.

*(Corpus caveat, measured not guessed: at n=20 the bootstrap CI half-width is ±0.10–0.20, so
**nothing under ~20 points is resolvable here.** The `.[data]` BEIR/HotpotQA swap is the single
highest-value next step — it would restore headroom and likely flip several verdicts.)*

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
