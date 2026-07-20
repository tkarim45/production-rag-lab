# Module: Evaluation deep-dive (Phase 11)

**Lesson.** Phases 0 and 10 both dead-ended on the same wall: **EM = 0.000 for every config**,
and token-F1 couldn't convincingly separate a real grounded LLM from a one-line extractive
baseline. Lexical metrics can't score a paraphrase. This phase builds the fix, a G-Eval-style
CoT LLM judge, plus faithfulness/context metrics that say *which half of the pipeline* is
broken, and a bootstrap CI so a 1-run delta isn't crowned a winner.

## Implemented
- `judge.py`, **G-Eval CoT correctness judge** (Claude Haiku/Bedrock, temp 0). Rubric 1 to 5 →
  [0,1]. Separate call from the generator; sees question + gold + candidate (not the context, 
  it scores *correctness*, not faithfulness).
- `faithfulness.py`, deps-free **groundedness** (fraction of the answer's content words present
  in the retrieved context, a lexical *lower bound*, honest about its blind spots), plus
  **context precision** (are retrieved chunks from relevant docs) and **context recall** (does
  the context even contain the gold answer).
- `significance.py`, from-scratch **paired bootstrap 95% CI** (10k resamples).

## Result: REAL Claude Haiku judge over the Phase 10 prompt styles (20 queries, judge cost $0.05)

| generator | EM | token-F1 | **JUDGE** | groundedness | ctx precision | ctx recall |
|---|--:|--:|--:|--:|--:|--:|
| extractive_mock | 0.000 | 0.232 | **0.650** | **1.000** | 0.400 | 0.960 |
| bare | 0.000 | 0.355 | **1.000** | 0.819 | 0.400 | 0.960 |
| grounded | 0.000 | 0.379 | **1.000** | 0.796 | 0.400 | 0.960 |
| cite_forced | 0.000 | 0.409 | **1.000** | **0.902** | 0.400 | 0.960 |
| abstain | 0.000 | 0.417 | **1.000** | 0.840 | 0.400 | 0.960 |

**Paired bootstrap 95% CI (10k resamples):**

| comparison | metric | delta | 95% CI | verdict |
|---|---|--:|---|---|
| bare − mock | **EM** | +0.000 | [+0.000, +0.000] | **not significant** |
| bare − mock | **judge** | **+0.350** | [+0.175, +0.550] | **SIGNIFICANT** |
| abstain − bare | **token-F1** | +0.062 | [+0.027, +0.102] | **SIGNIFICANT** |
| abstain − bare | **judge** | +0.000 | [+0.000, +0.000] | **not significant** |

## Honest findings

1. **EM is statistically proven useless here.** 0.000 across every config, CI [0.000, 0.000].
   It cannot distinguish a perfectly correct grounded answer from a mediocre extractive one.
   Phases 0 and 10 suspected it; the bootstrap makes it a measurement.

2. **The judge resolves exactly what EM could not:** bare − mock = **+0.350, CI [+0.175, +0.550],
   significant.** Real Claude is perfectly correct (1.000); the mock is 0.650. This is the
   phase's reason to exist, and it cost **$0.05**.

3. ⚠️ **The judge OVERTURNS Phase 10's headline, and Phase 10 was wrong.** Phase 10 reported
   prompt style as "the best ROI in the lab" (bare→abstain = +6.2 token-F1, +17%). The judge
   scores **all four real prompt styles at exactly 1.000**, abstain − bare = +0.000, CI
   [0.000, 0.000], **not significant**. Meanwhile token-F1 calls that same gap *significant*
   (+0.062, CI excludes 0). Both are right about what they measure: **the styles differ in
   phrasing overlap with the gold string, not in factual correctness.** Phase 10's F1 gain was
   lexical, not semantic. The corrected claim: *on this corpus, prompt style changes citation
   behaviour and verbosity, not correctness.* A significant lexical delta is not a quality
   delta, and this is precisely why "is there an eval framework here, or is it vibes?" is the
   question interviewers ask.

4. **Groundedness INVERTS the ranking, the trap in one row.** The extractive mock scores
   **1.000 groundedness** (it literally copies a context sentence, so it cannot be ungrounded)
   while being only **0.650 correct**. Real Claude paraphrases → 0.80 to 0.90 groundedness → but
   **1.000 correct**. *A groundedness metric alone would rank the mock above Claude.*
   Faithfulness ≠ correctness; ship both or you'll optimize for a copy-paste bot.

5. **`cite_forced` has the highest groundedness of the real styles (0.902 vs 0.796 to 0.840)**, 
   independent confirmation of Phase 10's mechanism: forcing citations really does keep the
   model closer to the context. The judge just says it doesn't make it *more correct* here.

6. **The judge saturates (1.000 × 4), no headroom, no signal.** The same meta-finding as every
   other phase: this corpus is too easy for the judge to discriminate. A judge that scores
   everything perfect measures nothing. Harder questions (or a distractor-heavy corpus) are the
   prerequisite for the judge to earn its cost.

7. **Context precision 0.400 / recall 0.960 is an actionable retrieval finding.** The context
   almost always contains the answer (recall 0.96) but 3 of the 5 chunks we stuff are irrelevant
   (precision 0.40). That's pure token waste, a smaller `final_k` would cut cost with no quality
   loss. Precision/recall split tells you *which* knob; the aggregate never would.

8. **The judge is uncalibrated and this repo says so.** Real calibration means judge-vs-human
   agreement (Cohen's κ) on a labeled sample. There are no human labels here, so the judge is
   reported *alongside* the lexical metrics and the divergence is the finding, rather than
   crowning the judge as ground truth. That gap is the honest open item, not a footnote.
