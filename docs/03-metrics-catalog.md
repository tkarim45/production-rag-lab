# 03 — Metrics catalog

Every metric the harness computes, defined once. All implemented in `harness/metrics/`,
unit-tested against hand-computed cases (Phase 0). Report all of them per config so
comparisons are apples-to-apples.

## Retrieval metrics (need relevance judgments / qrels)

| Metric | Definition | Notes |
|---|---|---|
| **Recall@k** | frac of relevant docs retrieved in top-k | primary "did we find it" |
| **Precision@k** | frac of top-k that are relevant | noise in the context window |
| **Hit-Rate@k** | 1 if ≥1 relevant in top-k else 0, averaged | lenient; good for QA |
| **MRR** | mean of 1/rank of first relevant | rewards ranking the answer high |
| **NDCG@k** | discounted cumulative gain vs ideal | graded relevance, position-aware |
| **MAP** | mean average precision across queries | overall ranking quality |
| **Context Precision** | are retrieved chunks relevant to the answer (RAGAS) | LLM-judged |
| **Context Recall** | is the gold answer covered by retrieved context (RAGAS) | LLM-judged |

## End-to-end answer metrics (need gold answers)

| Metric | Definition | Notes |
|---|---|---|
| **Exact Match (EM)** | normalized answer == gold | strict; under-counts good answers |
| **Token F1** | token overlap F1 with gold | partial credit |
| **Answer Correctness** | LLM-judged correctness vs gold | handles paraphrase |
| **Answer Relevancy** | does the answer address the question (RAGAS) | LLM-judged |

## Faithfulness / hallucination metrics

| Metric | Definition | Notes |
|---|---|---|
| **Groundedness / Faithfulness** | frac of answer claims supported by retrieved context | core RAG safety |
| **Hallucination Rate** | frac of answers with ≥1 unsupported claim | lower = better |
| **Citation Accuracy** | do cited chunks actually support the cited claim | for citing pipelines |
| **Abstention Correctness** | correctly says "I don't know" when context lacks answer | no-answer handling |

## Efficiency metrics (first-class, not afterthoughts)

| Metric | Definition |
|---|---|
| **Latency p50 / p95 / p99** | end-to-end and per-stage (retrieve, rerank, generate) |
| **TTFT** | time to first token (streaming) |
| **Cost / query** | $ per query (embedding + LLM tokens) |
| **Cost / correct answer** | cost/query ÷ correctness — the real efficiency number |
| **Index build time** | seconds to build the index |
| **Index memory / storage** | RAM + disk footprint |
| **Throughput (QPS)** | queries/sec at a concurrency level |

## Human-evaluation metrics

| Metric | Definition |
|---|---|
| **Human correctness / helpfulness** | rubric scores from human labelers |
| **Inter-annotator agreement** | Cohen's κ / Krippendorff's α across annotators |
| **Judge↔human agreement** | correlation of LLM-judge with human labels (calibration) |

## Drift / ops metrics

| Metric | Definition |
|---|---|
| **PSI** | population stability index on embedding/query distributions |
| **Retrieval-quality drift** | rolling Recall@k / groundedness over time |
| **Feedback signal** | thumbs-up rate, click-through on cited sources |

## Reporting rules

- Always report a metric **with its efficiency cost** — a +2pt correctness that triples
  latency/cost is a different decision than a free one.
- Report **variance / significance** when comparing configs (bootstrap CI or a proper
  test) — don't crown a winner on a 1-run delta.
- Split e2e metrics by **query type** (single-hop vs multi-hop) — aggregates hide the
  multi-hop failure of query-transform methods.
