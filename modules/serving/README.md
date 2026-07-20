# Module: Serving & performance (Phase 12)

**Lesson.** Every earlier phase optimized *quality*. Serving asks a different question: of the
milliseconds and dollars a request actually costs, where do they go, and what can you refuse
to compute twice? The answer is almost never where the engineering instinct points. Two caches
are built here (embedding, semantic) and neither is a free win: one is *net negative* on a
cheap embedder, and the other buys its hit rate with wrong answers. The report that tells you
which is which is the latency budget, so that ships too.

## What's implemented
`cached` (embedder wrapper), disk cache, key = `sha256(text)` inside a namespace of the
embedder's identity (name + params + dim + model id). Refuses corpus-fit bases.
`semantic_cached` (generator wrapper), serve a stored answer when cosine(query, cached) ≥
threshold; reports `cache_hit`/`cache_sim`/`cache_source_query_id` on every answer.
`python -m harness.latency_report <config>`, per-stage p50/p95 + share of budget.
`python -m harness.cache_report [--generator claude]`, the benchmark below.
`modules/serving/api.py`, optional FastAPI `/ask` + `/health` (`.[serve]`; degrades if absent).

## Result A: embedding cache, rebuild cost (`builtin_docs`, 25 chunks)

`python -m harness.cache_report`, cold = empty cache, warm = fresh wrapper on the populated
cache (what the next rebuild sees), both steady-state (torch warm-up burned before timing).

| base embedder | no cache | cold | warm | warm vs cold | **warm vs no cache** |
|---|--:|--:|--:|--:|--:|
| `hashing` (512d, FNV bag-of-words) | **0.9 ms** | 5.9 ms | 1.7 ms | 3.5× faster | **1.9× SLOWER** |
| `sentence_transformer` (bge-small, 384d) | 223.3 ms | 147.3 ms | **10.6 ms** | 13.9× faster | **21.1× faster** |

Warm hit rate 1.000 in both rows, the cache works perfectly in both. Only one of them is
worth having.

## Result B: semantic cache, hit rate vs false hits (real Claude Haiku 4.5 / Bedrock)

`python -m harness.cache_report --generator claude`, 40 requests = 20 distinct eval questions
replayed twice, keyed on bge-small vectors. **Pass 1 is 20 different questions, so every pass-1
hit is by construction a false hit.** Ground truth = the gold answer already on each query; a
hit sourced from an entry with a different gold answer is false. The cache never sees the
labels, only the audit does.

| threshold | hit rate | hits | true | **FALSE** | false/hits | p1 (false) | p2 (repeats) | cost $ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.99 | 50.0% | 20 | 20 | **0** | 0.0% | 0 | 20 | 0.0132 |
| 0.90 | 50.0% | 20 | 20 | **0** | 0.0% | 0 | 20 | 0.0132 |
| 0.80 | 50.0% | 20 | 20 | **0** | 0.0% | 0 | 20 | 0.0132 |
| 0.70 | 55.0% | 22 | 18 | **4** | 18.2% | 2 | 20 | 0.0120 |
| 0.60 | 65.0% | 26 | 14 | **12** | 46.2% | 6 | 20 | 0.0094 |

A hit costs **$0 and ~15 to 29 ms** (just re-embedding the query) against **~1300 to 1480 ms** for a
miss, **~44 to 90× faster**. The two questions that break first, at cosine 0.713 and 0.709:

> asked *"Which Roman emperor was the first, and in what year did the empire begin?"* →
> served the answer to *"When did the Western Roman Empire fall and who deposed the last emperor?"*
> asked *"What is released as a byproduct when water is split during photosynthesis?"* →
> served the answer to *"How do plants convert light energy into chemical energy?"*

## Result C: the latency budget (`builtin_mini`, n=12)

`python -m harness.latency_report configs/naive_claude.yaml`

| stage | p50 ms | p95 ms | mean ms | % of budget (real Haiku) | % of budget (mock generator) |
|---|--:|--:|--:|--:|--:|
| embed_query | 0.093 | 0.15 | 0.088 | 0.006% | 14.3% |
| retrieve | 0.071 | 0.15 | 0.084 | **0.006%** | 18.7% |
| assemble | 0.011 | 0.04 | 0.012 | 0.001% | 2.4% |
| generate | 1287.14 | 2178.41 | 1377.72 | **99.99%** | 64.6% |
| **total** | **1287.23** | **2178.61** | **1377.90** | 100% | 100% (total p50 **0.067 ms**) |

**Honest findings.**
1. **An embedding cache is only worth it if embedding is expensive, and "expensive" is not a
   given.** On `hashing`, a *perfectly working* cache at a 100% warm hit rate (1.7 ms) is still
   **slower than just recomputing from scratch** (0.9 ms): 25 disk reads lose to 25 hash loops.
   On bge-small the identical code is **21.1× faster** (10.6 ms vs 223.3 ms). Same cache, same
   corpus, opposite verdict, the *base embedder's cost* decides, and nothing else does. Note
   this is the same shape as the Phase 5/7/9 finding: a technique only helps where there's
   headroom.
2. **The cache gets cheaper by getting wronger, so never tune on hit rate or cost.** Both
   metrics improve monotonically as the threshold drops (50%→65% hits, $0.0132→$0.0094) and
   both are improving *because* the cache is serving other people's answers. At 0.60, **46% of
   every hit is wrong**. Hit rate argues for a lower threshold at every step; only the false-hit
   column ever says stop.
3. **A false hit is permanent, cache poisoning is real and it compounds.** At 0.70 the 2 pass-1
   false hits become 4: a query served a wrong answer never inserts its *own* correct answer, so
   every later repeat of it hits the same wrong entry forever. Errors here don't average out
   with volume, they replicate.
4. **Generation is 99.99% of the p50 budget; retrieval is 0.006%** (0.071 ms of 1287 ms). Every
   retrieval optimization in Phases 4 to 7 is *invisible* end-to-end, those phases buy quality,
   not latency, and the fastest possible retriever saves 0.07 ms off a 1.3-second request. Only
   a mock generator makes retrieval look worth optimizing (18.7% vs 0.006%, the mock flips the
   story entirely). This is also *why* the cache belongs in front of generation.
5. **The safe band is set by the embedding, not by the threshold you pick.** The highest cosine
   between two genuinely different questions in this set is **0.713** (bge-small), so any
   threshold ≥ ~0.72 takes every legitimate repeat at zero false hits, and the danger zone is
   narrow and knowable. **Where it can't be measured, say so:** keying on lexical `hashing`
   vectors instead (max distinct-question cosine 0.581) produced **zero false hits at every
   threshold tested**, but that is *not* evidence it's safer. This workload has no paraphrases,
   so the true hits a lexical key would **miss** are unmeasurable on this corpus, and a cache
   that never fires is trivially never wrong.
6. **A cache assumes the thing it caches is a pure function, `tfidf` isn't, so `cached`
   refuses it.** Its vectors depend on the corpus it was last fit on, so a cached vector would
   silently be from another corpus's statistics. Raising beats a plausible wrong number.

Caveats: 13-doc / 25-chunk corpus, 20 questions, n=12 to 40 requests, p95 is directional, the
sub-millisecond `hashing` timings move ±1 ms run to run (the *ordering* is stable, the digits
are not), and
`retrieve_ms` on a flat index of 25 vectors is a best case that would grow (though not to 1.3 s)
at real scale. The 50% hit rate is a property of the **workload** (exact repeats, no
paraphrases), not of the cache: read the false-hit cliff, not the hit rate. The semantic cache
wraps the *generator*, so a hit still pays retrieval, which finding 4 says costs nothing.
Prompt caching and TTFT/streaming, load shedding, and batched embed+rerank remain open.
