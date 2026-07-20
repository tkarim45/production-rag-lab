# Module: Generation (Phase 10)

**Lesson.** Retrieval can be perfect and the answer still bad. Generation is where grounding,
citation, abstention and decoding decide whether good context becomes a good answer, or a
confident hallucination. The prompt is the cheapest lever in the entire stack: it costs
nothing and ships in seconds.

## Implemented
`claude_prompted`, one Claude/Bedrock generator parameterized by **prompt style** and
**decoding**, so the phase sweeps them on the shared harness:
- `bare`, no grounding instruction (the control).
- `grounded`, answer only from the context.
- `cite_forced`, grounded + must cite `[n]`, refuse if uncited.
- `abstain`, grounded + explicit permission to say "I don't know".
- temperature / top_p knobs (`grounded_t07` = the same prompt at temp 0.7).

It also reports `has_citation` and `abstained` per answer, which the scorer turns into
**citation_rate** and **abstain_rate**.

## Result: REAL Claude Haiku 4.5 on AWS Bedrock, `builtin_docs` (20 queries), tfidf+flat

| generator | EM | token_f1 | citation rate | abstain rate | cost/query | p50 latency |
|---|--:|--:|--:|--:|--:|--:|
| extractive_mock (Phase 0 floor) | 0.000 | 0.232 |, |, | $0 | 0.1 ms |
| bare | 0.000 | 0.355 | 0.00 | 0.00 | $0.00069 | 1420 ms |
| grounded | 0.000 | 0.379 | **0.30** | 0.00 | $0.00066 | 1391 ms |
| cite_forced | 0.000 | 0.409 | **1.00** | 0.00 | $0.00066 | 1349 ms |
| **abstain** | 0.000 | **0.417** | 0.00 | 0.00 | **$0.00064** | 1244 ms |
| grounded @ temp 0.7 | 0.000 | 0.382 | 0.20 | 0.00 | $0.00065 | 1402 ms |

> [!warning] ⚠️ Finding #1 below was **overturned by Phase 11**, read both.
> Phase 11's LLM judge scores all four real prompt styles at **exactly 1.000** correctness
> (abstain − bare = +0.000, CI [0.000, 0.000], **not significant**), while token-F1 calls the
> same gap significant (+0.062, CI [+0.027, +0.102]). **The prompt styles differ in phrasing
> overlap with the gold string, not in factual correctness.** The +17% below is a *lexical*
> delta. Corrected claim: on this corpus, prompt style changes **citation behaviour and
> verbosity, not correctness**. Left standing, annotated, because the mistake is the lesson:
> a significant lexical delta is not a quality delta. See `modules/evaluation/README.md`.

**Honest findings.**
1. ~~**Prompt style is the best ROI in the whole lab.**~~ **(Overturned by Phase 11, see the
   warning above.)** `bare` → `abstain` is +6.2 token-F1 points (+17% relative) at identical cost
   (~$0.00065/query) and *lower* latency. The *cost* claim holds, prompt changes are free. The
   *quality* claim does not: the judge says all four styles are equally correct. What survives:
   prompt style is a free lever for **citation rate and groundedness** (findings 2 to 3), not for
   correctness on an easy corpus.
2. **Forcing citations makes the answer *better*, not just auditable** (0.355 → 0.409, +5.4
   pts). Citing forces the model to actually read the context instead of paraphrasing from
   memory. The audit trail is a side effect of the quality mechanism.
3. **Citation behaviour is emergent and promptable:** 0% (bare) → **30% spontaneously**
   (grounded) → 100% (forced). "Answer only from context" alone makes the model cite a third of
   the time unprompted.
4. **Permission to abstain didn't make it lazy.** `abstain` scored *highest* on F1 with a **0%
   abstention rate**, the context always contained the answer, and the model correctly never
   bailed. That's the good outcome: no false abstention. (This corpus can't test *true*
   abstention, every query is answerable. A no-answer eval set is the missing piece.)
5. **Temperature 0 vs 0.7 is a wash on quality** (0.379 vs 0.382, noise) but 0.7 costs you
   reproducibility. So the "temp 0 for eval" rule is about **determinism, not accuracy**, 
   exactly the nuance interviewers probe.
6. **EM = 0.000 for every real-LLM config**, while token-F1 ranges 0.36 to 0.42. Real grounded
   answers never string-match terse gold. Phase 0's warning, confirmed at scale: **EM is
   actively misleading for RAG**, it would rank all five prompt styles as identically worthless.
7. The mock floor (0.232) is 1.5 to 1.8× below the real model, the key-free baseline behaves as
   designed: a valid lower bound, not a substitute.
