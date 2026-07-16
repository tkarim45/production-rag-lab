# Module: Observability & ops (Phase 13)

**Lesson.** Every phase before this one asked "which config scores best?" This one asks the
two questions that come *after* you ship: **is it still working, and how would I know?** The
uncomfortable answer this phase produces is that the lab's own eval set is too small to
answer either — and that the honest response to "PSI says significant" or "config B scored
higher" is usually **"that number is noise."** The tools here are the ones that tell you so.

## Implemented
- `tracing.py` — per-request traces → SQLite (stdlib, no OTel/collector). Three tables:
  `traces` (query, answer, latency, tokens, cost), `spans` (per-stage latency), `retrievals`
  (rank, chunk_id, score). `TracedPipeline` wraps any `Pipeline` transparently; `Tracer.record()`
  is the runner-side hook. DB is gitignored (`*.sqlite3`) — traces are runtime state.
- `drift.py` — **from-scratch PSI**: `PSI = Σ (a% − e%)·ln(a%/e%)`, bands 0.10 / 0.25.
  Quantile bins from the reference, categorical for low-cardinality signals, epsilon-floored
  empty bins. Applied to query embeddings and to rolling retrieval quality.
- `ab.py` — **paired bootstrap 95% CI** on a metric delta between two configs (from scratch,
  numpy). Resamples query indices, same indices both arms.
- `gate.py` + `python -m harness.gate` — golden-threshold CI regression gate, `min` floors for
  quality and `max` ceilings for cost/latency, exit 0/1/2. Thresholds in `configs/golden_gate.yaml`.

---

## Result 1 — A/B with a bootstrap CI: which leaderboard deltas survive?

`python -m modules.ops.ab --vary embedder --a hashing --b tfidf`
(`builtin_docs`, n=20, paired percentile bootstrap, B=10,000)

| metric | A (hashing) | B (tfidf) | delta | 95% CI | verdict |
|---|--:|--:|--:|:--:|---|
| recall@1 | 0.550 | 0.950 | **+0.400** | [+0.200, +0.600] | **B wins** |
| recall@5 | 0.950 | 1.000 | +0.050 | [+0.000, +0.150] | inconclusive |
| mrr | 0.750 | 1.000 | **+0.250** | [+0.117, +0.400] | **B wins** |
| ndcg@10 | 0.797 | 1.000 | **+0.203** | [+0.091, +0.330] | **B wins** |
| token_f1 | 0.212 | 0.232 | +0.019 | [−0.005, +0.057] | inconclusive |

## Result 2 — the same tool, pointed at this repo's own Phase 9 headline

`python -m modules.ops.ab --vary chunker --a fixed --b contextual --pin embedder=hashing`

| metric | A (fixed) | B (contextual) | delta | 95% CI | verdict |
|---|--:|--:|--:|:--:|---|
| recall@1 | 0.550 | 0.650 | +0.100 | [+0.000, +0.250] | **inconclusive** |
| recall@5 | 0.950 | 0.950 | +0.000 | [+0.000, +0.000] | inconclusive |
| mrr | 0.750 | 0.817 | +0.067 | [−0.000, +0.158] | **inconclusive** |
| ndcg@10 | 0.797 | 0.847 | +0.050 | [−0.001, +0.118] | inconclusive |

Phase 9's README reports exactly these point estimates ("recall@1 **0.55→0.65 (+10pts)**, MRR
+6.7pts") and calls it a win. The point estimates reproduce. **The confidence does not.**

## Result 3 — PSI: the method works, and then the sample size destroys it

`python -m modules.ops.drift`

**(a) Method sanity — synthetic gaussians, n=10,000/sample.** PSI behaves exactly as advertised:

| case | PSI | verdict |
|---|--:|---|
| N(0,1) vs itself | 0.0000 | none |
| N(0,1) vs N(0,1) resampled | 0.0012 | none |
| N(0,1) vs N(0.5,1) — mean shift | 0.2367 | moderate |
| N(0,1) vs N(0,2) — variance shift | 0.4952 | significant |

**(b) The noise floor — split-half of ONE population, where no drift exists.** Any value above
zero is pure sampling noise. Median of 25 splits:

| n per half | bins=5 | bins=10 |
|--:|--:|--:|
| 10 | 2.6897 | 23.0256 |
| **20** ← this lab's eval set | **0.3610** | **2.6538** |
| 50 | 0.1547 | 0.3770 |
| 100 | 0.0814 | 0.1813 |
| 500 | 0.0148 | 0.0366 |
| 1,000 | 0.0078 | 0.0220 |
| 10,000 | 0.0005 | 0.0018 |

**(c) Real query-embedding drift — frozen tfidf embedder (dim=600), the queries move:**

| case | PSI(centroid-cos) | PSI(per-dim) | verdict |
|---|--:|--:|---|
| identical set vs itself (n=20) | **0.0000** | 0.0000 | none |
| split-half, **same** population (n=10 vs 10) | 23.0256 | 0.2938 | significant ⚠️ |
| `builtin_docs` vs `builtin_mini` queries (n=20 vs 12) | **6.8436** | 0.2131 | significant |

**(d) Retrieval-quality drift — per-query recall@5, tfidf → hashing:** reference mean 1.000 →
actual 0.950 (delta −0.050, i.e. **one query of twenty** flipped). PSI = **0.5435 → "significant"**.

## Result 4 — CI regression gate, red and green

`python -m harness.gate --golden configs/golden_gate.yaml` → **exit 0**

```
[PASS] mrr 1.0000 >= 0.9000 · [PASS] recall@1 0.9583 >= 0.8500 · [PASS] recall@5 1.0000 >= 0.9000
[PASS] token_f1 0.3284 >= 0.2800 · [PASS] cost_per_query_usd 0.0000 <= 0.0010
[PASS] latency_p50_ms 0.0648 <= 5.0000
GREEN — 6/6 checks passed. exit 0
```

`python -m harness.gate configs/naive.yaml --golden configs/golden_gate.yaml --dataset builtin_docs`
→ **exit 1** (same pipeline, same bar, the harder corpus):

| check | value | bar | |
|---|--:|--:|---|
| recall@1 | 0.7000 | ≥ 0.85 | **FAIL** |
| mrr | 0.8125 | ≥ 0.90 | **FAIL** |
| token_f1 | 0.2058 | ≥ 0.28 | **FAIL** |
| recall@5 | 0.9000 | ≥ 0.90 | pass |
| latency_p50_ms | 0.1391 | ≤ 5.0 | pass |
| cost_per_query_usd | 0.0000 | ≤ 0.001 | pass |

## Result 5 — tracing (`python -m modules.ops.tracing --config configs/naive.yaml`)

12 requests, p50/p95 = 0.08 / 0.29 ms, 1,686 in / 199 out tokens, $0.000000/query. Stage
budget from the `spans` table: `generate` 58.8% · `retrieve` 23.0% · `embed_query` 15.7% ·
`assemble` 2.5%. Every trace drills down to its ranked retrieval set with scores
(`#1 d10::0 score=0.6455 …`).

---

## Honest findings

1. **The 20-query eval set cannot resolve anything under ~20 points, and most of this repo's
   findings are under 20 points.** The bootstrap CI half-width is ±0.10–0.20 on retrieval
   metrics. It cleanly confirms tfidf over hashing (+40 pts recall@1, CI [+0.20, +0.60]) — a
   delta that large is real. It cannot confirm +5.0 pts on recall@5, or +1.9 on token_f1.
   Every phase in this lab that crowned a winner on a single-digit delta was reporting a point
   estimate, not a result.
2. **Applied to Phase 9, this phase's own tool says "inconclusive."** Contextual retrieval's
   +10 pt recall@1 lift has a 95% CI of [+0.000, +0.250] — it touches zero. This does **not**
   mean contextual retrieval doesn't work: the point estimate is unchanged, the direction is
   positive on every metric, and it matches Anthropic's published 35/49/67% reductions on real
   corpora. It means **this corpus cannot tell you that**, and Phase 9's confident framing
   outran its evidence. The finding is the epistemics, not a reversal.
3. **PSI's headline number at n=20 is measuring sample size, not drift.** Split-half of a
   *single* population — where drift is impossible by construction — scores **2.65** at
   bins=10, over 10× the 0.25 "significant" threshold. On the real query embeddings the noise
   floor (23.03) is **3.4× larger than the actual out-of-domain shift** (6.84): the noise is
   bigger than the signal, and the ordering is inverted. Any PSI monitor wired to this eval
   set would page someone every day and be muted within a week.
4. **PSI needs ~10 observations per bin, and that's the whole story.** The noise floor only
   drops under 0.10 at n≈100 (bins=5) or n≈500 (bins=10). The fix at small n is fewer bins —
   bins=5 cuts the n=20 floor from 2.65 to 0.36 — but 0.36 is *still* above the "significant"
   band. There is no binning that rescues n=20. The honest statement is not "PSI is broken";
   it's "**PSI is a large-sample statistic and 20 queries is not a large sample.**"
5. **At small n, PSI's magnitude is set by a constant we chose.** On the same drift-free n=10
   data, PSI reads 4.14 / 13.80 / 23.03 / 32.24 as the empty-bin epsilon goes 1e-2 → 1e-8 —
   and "significant" at all four. The number is an artifact of an implementation detail nobody
   puts in the dashboard. A PSI without its epsilon and bin count declared is not a number.
6. **Per-dim PSI is useless on sparse lexical embeddings — and it lies with a straight face.**
   On tfidf (dim=600, mostly zeros) it scored the *same* population as drifting **more**
   (0.2938) than a genuinely out-of-domain one (0.2131). Most dimensions are 0 in both
   populations, so they contribute nothing and dilute the few that matter. The centroid-cosine
   projection at least ordered them correctly. Neither is trustworthy at this n; they're shipped
   together because their disagreement is the tell.
7. **PSI on a saturated metric is a hair trigger.** Reference recall@5 is 1.000 for all 20
   queries — one distinct value. One query flipping (−0.05) produces PSI 0.5435, "significant,"
   because a perfect reference distribution makes *any* deviation infinitely surprising modulo
   the epsilon floor. On a near-binary outcome PSI carries **no more information than the mean
   delta** — and the mean delta at least comes with a CI. Monitor the rate; keep PSI for the
   input distribution where it has bins to work with.
8. **The gate is the one honest artifact here, precisely because it's crude.** It doesn't
   claim significance — it asserts a floor and blocks a merge. It catches *breakage*
   (recall@1 0.958 → 0.700, three checks red, exit 1), not *regression*: a real 2-point drop
   passes silently, and at n=12 one query flipping moves recall@1 by 0.083, which is why the
   margin exists. A gate that can only catch breakage is worth shipping; a gate tuned tight
   enough to catch 2 points at this n would be red on noise every third build and disabled by
   Friday.
9. **Latency's gate is deliberately loose (5 ms against a measured 0.065 ms) and that's the
   correct engineering call.** CI runners are shared and noisy; a flaky gate gets deleted,
   which is strictly worse than a loose one. It's sized to catch an accidental O(n²) or a real
   network call, not 2× jitter.
10. **The trace stage-breakdown is real but unrepresentative — and Phase 12 proves it.**
    `generate` is 58.8% of a 0.08 ms mock request, making `retrieve` (23.0%) look worth
    optimizing. Phase 12 measured the same budget on real Claude Haiku: **generation is 99.99%
    (1287 of 1287.2 ms), retrieval 0.006%.** The tracing *schema* is what transfers; these
    percentages are an artifact of a free generator. Trace tables are still worth building
    before you need them — the cost of not having the retrieval set for the request that went
    wrong is that the incident is unanswerable.
11. **Two independent bootstraps agree, which is why the number is trustworthy.** Phase 11's
    `significance.paired_delta_ci` resamples the difference vector; this module resamples
    shared query indices across both arms. Same estimator by algebra, different code, different
    RNG — they land within 0.02. That agreement is pinned as a test.

**Caveat, stated plainly: this phase's own conclusions are limited by the same n=20 it
criticizes.** The bootstrap CIs are honest about their width, but the noise-floor curve is
synthetic gaussians (chosen so "no drift" is true by construction — the only way to measure a
floor), and the drift verdicts on real embeddings are directional at best. What this corpus
**cannot** show: whether PSI would catch a *gradual* production drift (needs a time series, not
two snapshots), whether the gate's thresholds hold across machines (latency is laptop-measured),
whether query drift actually predicts quality drift (needs enough queries for both to be
resolvable — the 13-doc corpus makes retrieval saturate at recall@5 = 1.000, so there is no
quality signal left to drift), and any feedback-loop claim (thumbs → hard-negative mining needs
real users). The BEIR swap (Phase 1's deferred item) is what turns every number above from a
demonstration into a measurement.
