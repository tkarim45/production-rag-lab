"""Evaluation deep-dive (Phase 11).

Phases 0 and 10 both ended on the same wall: **EM = 0.000 and token-F1 barely separates a
real grounded LLM from a one-line extractive baseline.** Lexical metrics can't score a
paraphrase. This phase builds the fix — a G-Eval-style CoT LLM judge — plus the faithfulness
and context metrics that tell you *which half of the pipeline* is broken, and the bootstrap
CI that stops a 1-run delta from being crowned a winner.

Exports:
- `judge` — G-Eval CoT correctness judge (Claude/Bedrock, temp 0), scores 1-5 → [0,1].
- `faithfulness` — deps-free claim-support proxy: what fraction of answer content words are
  grounded in the retrieved context (the cheap, honest version of groundedness).
- `context_metrics` — context precision/recall at the doc level.
- `significance` — from-scratch bootstrap 95% CI for a metric delta.
"""

from modules.evaluation import faithfulness, judge, significance  # noqa: F401
