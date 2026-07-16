# 01 — Curriculum map (the full taxonomy)

Every technique the repo covers, grouped by layer. This is the "did we miss anything"
reference. Each leaf maps to a module under `modules/` with a lesson + benchmark. Items
marked ➕ were added beyond the original brief (production essentials).

---

## A. Data ingestion & parsing
- Format parsing: PDF (text + layout), HTML, DOCX, Markdown, CSV/tables, code ➕
- Layout-aware extraction: tables, columns, headers/footers, figures ➕
- OCR for scanned docs ➕
- Cleaning & normalization: boilerplate strip, whitespace, encoding, language detection ➕
- Deduplication: exact (hash) + near-dup (MinHash/SimHash) ➕
- Metadata enrichment: source, section, date, author, page, title path ➕

## B. Chunking
- Fixed-size (token/char) + overlap sweep
- Recursive character splitting
- Sentence / paragraph splitting
- Semantic chunking (embedding-similarity boundaries)
- Structural / layout-aware (headings, sections, tables)
- Proposition / atomic-fact chunking
- Parent-child / small-to-big (retrieve small, feed big)
- Late chunking (embed doc, then chunk embeddings) ➕
- Chunk-size × overlap benchmark sweep

## C. Embeddings ➕ (mostly added)
- Model selection via MTEB; domain fit
- Dimensionality tradeoff; Matryoshka truncation
- Quantization: int8, binary embeddings (recall vs memory/speed)
- Domain fine-tuning of the embedder (MLX-LoRA)
- Embedding cache (avoid recompute)
- Symmetric vs asymmetric (query vs passage) encoders; instruction-prefixed embedders

## D. Indexing & vector stores
- Exact: Flat (brute force) — ground-truth baseline
- ANN: IVF, IVFPQ, HNSW, ScaNN-style, DiskANN (on-disk) ➕
- Recall@k vs latency vs memory vs build-time Pareto (reuse `vector-db-benchmark` method)
- Sparse: BM25 (keyword)
- Learned sparse: SPLADE ➕
- Index build params tuning (nlist/nprobe, M/efConstruction/efSearch)

## E. Retrieval
- Dense (vector) retrieval
- Sparse (BM25) retrieval
- Hybrid: dense + sparse ➕
- Fusion: Reciprocal Rank Fusion, weighted score fusion ➕
- Metadata filtering (pre-filter vs post-filter; filter + ANN interaction)
- Diversity / redundancy control: MMR (Maximal Marginal Relevance)
- Top-k tuning

## F. Query understanding ➕ (mostly added)
- Query rewriting / normalization
- Query expansion (synonyms, PRF)
- HyDE (hypothetical document embeddings)
- Spelling correction
- Query classification / routing (which index / which strategy)
- Query decomposition (break complex → sub-queries)
- Multi-hop retrieval (chain retrievals across bridge docs)
- Iterative / recursive retrieval (retrieve → read → retrieve again)
- Step-back prompting

## G. Reranking
- Cross-encoder reranker
- Late-interaction (ColBERT) ➕
- LLM reranker (pointwise / pairwise / listwise) ➕
- Reranker fusion
- Reranker latency vs quality tradeoff; when NOT to rerank

## H. Context assembly
- Selecting best chunks (score threshold vs top-k)
- Deduplicating similar retrieved chunks ➕ (clustering near-dups)
- Clustering retrieved chunks (group + summarize) ➕
- Ordering: lost-in-the-middle → reorder most-relevant to edges
- Context compression / summarization (LLMLingua-style) ➕
- Context-window budgeting (fit k chunks under a token budget, cost-aware)
- Citation-preserving assembly

## I. Advanced retrieval flows
- GraphRAG (entity/knowledge graph)
- RAPTOR (hierarchical recursive summary tree)
- Contextual retrieval (Anthropic — prepend chunk context before embedding) ➕
- Self-RAG (model decides when/what to retrieve) ➕
- Corrective RAG / CRAG (grade retrieval, fall back to web) ➕
- Adaptive RAG (route by query complexity) ➕
- Agentic RAG (tool-using retrieval agent) ➕

## J. Generation
- Grounding & citation prompts (force cite, refuse if uncited)
- Prompt engineering for RAG (system prompt, few-shot, format)
- Temperature / decoding params (temp, top-p, top-k) sweep
- The LLM harness (provider abstraction, temp 0 for eval, retries)
- Abstention / no-answer / "I don't know" + confidence ➕
- Answer formatting / structured output

## K. Evaluation
- **Retrieval metrics**: Recall@k, Precision@k, MRR, Hit-Rate, NDCG@k, MAP
- **End-to-end**: Exact Match (EM), token F1, answer correctness
- **Faithfulness**: groundedness, hallucination rate, context-precision, context-recall
- **RAGAS** suite integration
- **LLM-as-judge**: G-Eval CoT, calibrated against human labels ➕
- **Human evaluation**: labeling interface, inter-annotator agreement (Cohen's κ) ➕
- **Synthetic eval-set generation** (question/gold-chunk/gold-answer) + contamination check ➕
- **Cost/latency as first-class metrics** ➕
- Statistical significance of config comparisons ➕

## L. Serving & performance ➕ (mostly added)
- FastAPI async serving; streaming + TTFT
- Semantic cache (reuse `semantic-cache` repo) + prompt cache + embedding cache
- Latency budgeting across stages
- Cost / token economics (per-query $, per-correct-answer $)
- Load shedding / concurrency limits
- Batching embeddings & rerank

## M. Observability & ops ➕ (mostly added)
- Per-request tracing (retrieval set, scores, tokens, latency, cost)
- Monitoring dashboards
- Drift: embedding drift, query-distribution drift, retrieval-quality drift (PSI)
- A/B testing RAG configs online
- Feedback loops (thumbs/click → hard-negative mining)
- CI regression gate (golden eval gates merges)

## N. Security & governance ➕ (mostly added)
- Prompt injection via retrieved documents / data poisoning defense
- PII detection & redaction in ingested docs and outputs
- Permission-aware / access-controlled retrieval (row/doc-level ACLs)
- Multi-tenancy isolation
- Audit logging
- Right-to-be-forgotten: hard-delete a doc from index + caches
- Content provenance / source trust scoring

## O. Scaling & maintenance ➕ (mostly added)
- Scaling to 1M+ documents (approximate + on-disk)
- Sharding / distributed index
- Incremental indexing / real-time updates (CDC)
- Index compaction, versioning, blue-green reindex
- Storage tiering (hot memory ↔ warm disk ↔ cold object store)
- Reindexing strategy on embedder change
- Chunk redundancy / storage-cost management

## P. Optional advanced tracks
- Multimodal RAG (images, tables, charts; vision embeddings) ➕
- Multilingual / cross-lingual RAG ➕
- Long-context vs RAG tradeoff study ➕
- Structured data: SQL + RAG hybrid (text-to-SQL over retrieved schema) ➕

---

Every item above has a home in the phased roadmap (`02-roadmap.md`) and a checkbox in
[`02-roadmap.md`](02-roadmap.md). If a technique isn't listed here, it isn't in scope — add it here first.
