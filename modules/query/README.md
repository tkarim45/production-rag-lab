# Module: Query understanding (Phase 6)

**Lesson.** The query the user types is often not the query that retrieves best. Query
understanding rewrites/expands/decomposes it before retrieval: PRF and multi-query for recall,
HyDE for zero-shot dense, decomposition/multi-hop for compositional questions. Every transform
costs an extra retrieval (and often an LLM call) — so it must earn its latency.

## Implemented
- `prf` — pseudo-relevance feedback (deps-free, deterministic): retrieve once, harvest the top
  non-stopword terms from the top docs, append to the query, retrieve again.
- `multiquery_prf` — retrieve for `[original, PRF-expanded]` and RRF-fuse (keep the original's
  precision + the expansion's recall).
- `hyde` — real Claude/Bedrock: generate a hypothetical answer sentence and retrieve with it
  (its vocabulary matches target passages better than a short question). Optional (needs creds).

## Result — `builtin_docs`, tfidf embed

`python -m harness.sweep --vary query_transformer --options none prf multiquery_prf`

| transformer | recall@1 | recall@5 | MRR | p50 latency |
|---|--:|--:|--:|--:|
| none | **0.950** | 1.000 | **1.000** | 0.11 ms |
| prf | 0.900 | 1.000 | 0.975 | 0.22 ms |
| multiquery_prf | 0.900 | 1.000 | 0.975 | 0.20 ms |

**Honest findings.**
1. **PRF *hurts* here (recall@1 0.95 → 0.90, MRR 1.00 → 0.975) at 2× latency.** When retrieval
   is already good, expansion terms harvested from the top docs add noise that demotes the true
   #1 — classic query-drift. Query expansion is a fix for *under*-retrieval (vocabulary
   mismatch, short queries), not a free upgrade. On a saturated corpus it's strictly negative.
   This is the counter-intuitive, honest result the harness is built to surface.
2. **Multi-query (fuse original + expanded) recovers some of PRF's damage** but still doesn't
   beat the baseline here — the original query was already optimal.
3. **Multi-hop needs structure, not rephrasing.** The built-in corpus has only 2 multi-hop
   queries and they're easy enough that all methods hit 1.000, so this corpus can't *show* the
   failure — but the mechanism is the known one (documented across the vault + `rag-architectures`):
   a query transform can't retrieve a bridge document it has no words for; you need iterative
   retrieval or decomposition. Demonstrating it needs a HotpotQA-style set (the `.[data]` swap).

Takeaway for interviews: "would you add query expansion?" → "only after measuring that
retrieval is *under*-performing on the query type; on well-served queries it drifts and costs
latency." That nuance is the senior signal.
