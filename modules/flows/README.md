# Module: Advanced flows: Contextual Retrieval (Phase 9)

**Lesson.** Chunks are ambiguous in isolation: *"Its revenue grew 3%"* is unretrievable because
"Its" refers to a company named twenty chunks earlier. **Contextual Retrieval** (Anthropic,
Sept 2024) fixes this at the cheapest possible point, prepend a short situating blurb to each
chunk **before** embedding and BM25-indexing it. Reported retrieval-failure reduction: 35%
(contextual embeddings) → 49% (+ contextual BM25) → 67% (+ reranking).

## Implemented
- `contextual`, deps-free: prefix the document title/section (the structural context ingestion
  already gives you). Deterministic, zero LLM cost.
- `contextual_llm`, the paper's method: Claude writes a one-sentence blurb per chunk (optional;
  prompt caching is what makes per-chunk contextualization affordable in production).

Both are **chunker wrappers**, the context is baked into `chunk.text` before the embedder sees
it, which is exactly where the paper puts it. `raw_text` is preserved in metadata.

## Result: `builtin_docs`, matched chunk params (size 60 / overlap 10, 25 chunks both arms)

**A) On a WEAK first stage** (`hashing` embedder, where retrieval has headroom):

`python -m harness.sweep --vary chunker --embedder hashing --options fixed contextual`

| chunker | recall@1 | recall@5 | MRR | NDCG@10 | MAP |
|---|--:|--:|--:|--:|--:|
| fixed | 0.550 | 0.950 | 0.750 | 0.797 | 0.742 |
| **contextual** | **0.650** | 0.950 | **0.817** | **0.847** | **0.808** |

**+10pts recall@1, +6.7pts MRR, +5pts NDCG**, same chunk count, negligible latency.

**B) On a SATURATED stage** (`tfidf`): 0.950 → 0.950, MRR 1.000 → 1.000. **No change.**

> [!warning] ⚠️ Finding #1 was **qualified by Phase 13**, the point estimate is real, the claim isn't licensed.
> Phase 13's paired bootstrap reproduces the +10pt recall@1 lift exactly as a point estimate,
> but the **95% CI is [+0.000, +0.250]**, it touches zero. At n=20 the CI half-width is
> ±0.10 to 0.20, so **nothing under ~20 points is statistically resolvable on this corpus.** The
> direction is right and matches Anthropic's published result; this eval set simply cannot
> license the number. Honest status: **promising, not proven.** See `modules/ops/README.md`.

**Honest findings.**
1. **Contextual retrieval works, and it's nearly free** in the deps-free form, prefixing the
   title lifted weak-stage recall@1 by 10 points with no extra chunks and no LLM call. The
   direction matches Anthropic's result; the magnitude is corpus- and stage-dependent.
   **Caveat (Phase 13): CI [+0.000, +0.250], directional, not significant at n=20.**
2. **It only helps where retrieval has headroom** (the recurring theme of Phases 5 to 9). On a
   saturated stage the prefix changes nothing, you cannot improve a 1.000.
3. **The cheap version captured real value.** The paper uses an LLM per chunk; here the
   *structural* context (title), free, already in the ingestion metadata, did the work. Try
   the free structural context before paying for LLM contextualization; measure the gap.
4. Deferred honestly: Self-RAG / CRAG / Adaptive / Agentic RAG need an LLM control loop **and**
   a corpus where retrieval fails often enough to route around. On a 13-doc set where dense
   already scores 1.000, they'd be measuring noise. See `rag-architectures` for those flows
   benchmarked on a harder corpus.
