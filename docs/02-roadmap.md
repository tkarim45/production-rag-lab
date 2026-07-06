# 02 — Roadmap (phased build plan)

Build **Phase 0 first** — it's the shared harness every other phase plugs into. After
that, phases are mostly independent; the suggested order builds the linear RAG pipeline
front-to-back, then wraps production concerns around it. Each phase is a shippable
mini-project with its own results table.

Rule: **every technique is measured on the same harness (Phase 0) against the same corpus
+ eval set.** No phase is "done" without a committed results table and a short honest
writeup (including where the fancy option loses).

Rough calendar assumes solo, part-time. `➕` = production essential added beyond the brief.

---

## Phase 0 — Harness, datasets, metrics, naive baseline (Weeks 1–4) — FOUNDATION
The whole repo depends on this. Do not skip.
- Repo skeleton (`harness/ modules/ configs/ data/ results/`), Makefile, pytest, ruff.
- **Pipeline contract**: a config-driven pipeline (`ingest → chunk → embed → index →
  retrieve → rerank → assemble → generate → evaluate`) where each stage is a swappable
  component with a fixed interface.
- **Dataset loaders**: BEIR subset (qrels) + a QA set (gold answers) + a domain corpus.
- **Metrics module** (all of `03-metrics-catalog.md`), unit-tested against known cases.
- **Runner + leaderboard**: run any config → emit metrics JSON → append to a leaderboard.
- **Naive baseline RAG** end-to-end (fixed chunk + dense + top-k + Claude) with full
  metrics. This is the number every later phase must beat or justify.

**Deliverable:** `make bench configs/naive.yaml` → metrics JSON + leaderboard row.

## Phase 1 — Ingestion & parsing (Weeks 5–7)
- Parsers (PDF/HTML/DOCX/tables/OCR), cleaning, normalization, dedup (exact + near-dup),
  metadata enrichment. Measure: parse fidelity, dedup rate, doc count before/after.

**Deliverable:** clean, deduped, metadata-rich corpus + an ingestion quality report.

## Phase 2 — Chunking (Weeks 8–11)
- Implement all strategies (§B). Benchmark each on retrieval + e2e metrics; run the
  chunk-size × overlap sweep. Honest finding on semantic vs recursive.

**Deliverable:** chunking leaderboard (retrieval quality vs chunk count/storage).

## Phase 3 — Embeddings ➕ (Weeks 12–15)
- Model selection (MTEB), dim/Matryoshka, int8/binary quantization, embedder fine-tune
  (MLX-LoRA), embedding cache. Measure: retrieval quality vs memory vs speed.

**Deliverable:** embedding leaderboard incl. quantization recall/memory tradeoff.

## Phase 4 — Indexing & vector stores (Weeks 16–19)
- Flat/IVF/IVFPQ/HNSW/DiskANN + BM25 + SPLADE. Recall@k vs latency vs memory vs build
  Pareto (reuse `vector-db-benchmark`). Param tuning.

**Deliverable:** ANN Pareto chart + index-choice guide.

## Phase 5 — Retrieval (Weeks 20–23)
- Dense, sparse, hybrid + fusion (RRF/weighted), metadata filtering (pre/post), MMR.
- Top-k tuning. Honest hybrid-vs-dense finding.

**Deliverable:** retrieval-strategy leaderboard.

## Phase 6 — Query understanding ➕ (Weeks 24–28)
- Rewriting, expansion, HyDE, spelling, classification/routing, **decomposition, multi-hop,
  iterative retrieval**, step-back. Show query-transform helps single-hop, may fail
  multi-hop (needs structural methods).

**Deliverable:** query-understanding leaderboard split by single-hop vs multi-hop.

## Phase 7 — Reranking (Weeks 29–32)
- Cross-encoder, ColBERT/late-interaction, LLM reranker (pointwise/listwise), fusion.
- Latency vs quality; when NOT to rerank.

**Deliverable:** rerank leaderboard with latency cost per quality point.

## Phase 8 — Context assembly (Weeks 33–36)
- Best-chunk selection, **dedup/cluster similar chunks**, lost-in-the-middle reordering,
  context compression, token-budgeted assembly, citation preservation.

**Deliverable:** context-assembly leaderboard (answer quality vs tokens/cost).

## Phase 9 — Advanced retrieval flows (Weeks 37–43)
- GraphRAG, RAPTOR, **contextual retrieval**, Self-RAG, CRAG, Adaptive RAG, Agentic RAG.
- Each as a full pipeline over the shared eval. Which flow wins which query type.

**Deliverable:** advanced-flow leaderboard + retrieval-failure-rate reduction numbers.

## Phase 10 — Generation (Weeks 44–47)
- Grounding/citation prompts, prompt engineering, **temperature/decoding sweep**, LLM
  harness, abstention/no-answer + confidence, structured output.

**Deliverable:** generation study (hallucination rate vs temperature; abstention accuracy).

## Phase 11 — Evaluation deep-dive (Weeks 48–53)
- Full retrieval + e2e metric suite, RAGAS, **calibrated LLM-judge**, **human-eval UI +
  inter-annotator agreement**, synthetic eval-set generation + contamination check,
  significance testing.

**Deliverable:** the evaluation toolkit + a judge-calibration report.

## Phase 12 — Serving & performance ➕ (Weeks 54–58)
- FastAPI async + streaming/TTFT, semantic + prompt + embedding caching, latency
  budgeting, **cost/token economics (per-query & per-correct-answer $)**, load shedding.

**Deliverable:** served RAG API + a cost/latency report.

## Phase 13 — Observability & ops ➕ (Weeks 59–63)
- Tracing, dashboards, **drift monitors (embedding/query/quality PSI)**, **A/B config
  testing**, **feedback loops (hard-negative mining)**, **CI regression gate**.

**Deliverable:** ops dashboards + a red/green CI gate demo.

## Phase 14 — Security & governance ➕ (Weeks 64–68)
- **Prompt-injection-via-document / data-poisoning defense**, PII redaction,
  **permission-aware retrieval (ACLs)**, multi-tenancy, audit logs,
  **right-to-be-forgotten (delete from index + caches)**, source-trust scoring.

**Deliverable:** security test suite (injection blocked, ACL enforced, doc fully deletable).

## Phase 15 — Scaling & maintenance ➕ (Weeks 69–74)
- **Million-doc** run (approximate + on-disk/DiskANN), sharding/distributed,
  **incremental/real-time indexing (CDC)**, compaction, index versioning, blue-green
  reindex, storage tiering, reindex-on-embedder-change.

**Deliverable:** scaling report (latency/memory/storage at 1M docs) — big run is an
optional cloud burst; laptop proves the code path on a smaller corpus.

## Phase 16 — Capstone integration & leaderboard (Weeks 75–80) — HEADLINE
- Compose the best-of-every-layer into **one configurable production RAG**.
- Produce the **cross-layer leaderboard** and a final report: every decision, the number
  that justifies it, and the honest "dataset-dependent" caveats.
- Deploy the demo + a UI showing a query flowing through every instrumented stage.

**Deliverable:** the composed pipeline + master leaderboard + report + recording.

---

## Optional tracks (any time after Phase 11)
- **T1 Multimodal RAG** — images/tables/charts, vision embeddings.
- **T2 Multilingual RAG** — cross-lingual retrieval + eval.
- **T3 Long-context vs RAG** — when does stuffing the context beat retrieval.
- **T4 SQL + RAG hybrid** — structured + unstructured.

---

## Dependency graph

```
Phase 0 (harness) ─▶ everything
1 ingest ─▶ 2 chunk ─▶ 3 embed ─▶ 4 index ─▶ 5 retrieve ─▶ 6 query ─▶ 7 rerank ─▶ 8 context ─▶ 10 generate
                                                       └─▶ 9 advanced flows (uses 2–8)
11 eval  (deepens Phase 0's metrics; used by all)
12 serve ─▶ 13 ops ─▶ 14 security ─▶ 15 scale ─▶ 16 capstone
```
