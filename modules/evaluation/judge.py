"""LLM-as-judge (Phase 11) — the fix for the EM/F1 blind spot Phases 0 and 10 exposed.

G-Eval style: give the judge the rubric, make it reason step by step, then emit a 1-5 score
which we normalize to [0,1]. Temperature 0 for reproducibility. The judge is a *different
call* from the generator (never let a model grade its own answer in the same breath), and it
sees the question, the gold answer, and the candidate — not the context — because we're
scoring answer correctness, not faithfulness.

Honest caveat baked into the module: an uncalibrated judge is an opinion. Real calibration
means measuring judge-vs-human agreement (Cohen's κ) on a labeled sample. That needs human
labels this repo doesn't have — so we report the judge's scores AND the lexical metrics side
by side, and let the divergence be the finding, rather than claiming the judge is ground truth.
"""

from __future__ import annotations

import os
import re

_RUBRIC = """You are grading a question-answering system.

Score the CANDIDATE answer against the GOLD answer for factual correctness on a 1-5 scale:
5 = fully correct; states the same facts as the gold (wording may differ completely)
4 = correct but missing a minor detail present in the gold
3 = partially correct; some right facts, some missing or vague
2 = mostly wrong; only incidental overlap
1 = wrong, or refuses/says it doesn't know when the gold has an answer

Paraphrase is NOT penalised. Extra correct detail is NOT penalised. Only factual agreement
with the gold matters.

Think step by step in one short sentence, then output the score on its own final line as:
SCORE: <n>"""


class ClaudeJudge:  # pragma: no cover - needs creds
    """G-Eval CoT correctness judge on Claude/Bedrock."""

    def __init__(
        self,
        model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str | None = None,
        max_tokens: int = 200,
        price_in_per_mtok: float = 1.00,
        price_out_per_mtok: float = 5.00,
    ):
        from anthropic import AnthropicBedrock

        region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self._c = AnthropicBedrock(aws_region=region)
        self.model, self.max_tokens = model, max_tokens
        self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok
        self.cost_usd = 0.0

    def score_one(self, question: str, gold: str, candidate: str) -> float:
        """Return correctness in [0,1] (from the 1-5 rubric). 0.0 if unparseable."""
        msg = self._c.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0, system=_RUBRIC,
            messages=[{"role": "user", "content":
                       f"QUESTION: {question}\n\nGOLD: {gold}\n\nCANDIDATE: {candidate}"}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        self.cost_usd += (msg.usage.input_tokens / 1e6 * self.price_in
                          + msg.usage.output_tokens / 1e6 * self.price_out)
        m = re.search(r"SCORE:\s*([1-5])", text)
        if not m:
            m = re.search(r"\b([1-5])\b", text[::-1])   # last digit fallback
            if not m:
                return 0.0
        n = int(m.group(1))
        return (n - 1) / 4.0      # 1..5 → 0..1


def judge_batch(results, judge: "ClaudeJudge") -> dict[str, float]:  # pragma: no cover
    """Score a batch of PipelineResults that carry gold answers."""
    graded = [r for r in results if r.query.gold_answer]
    if not graded:
        return {}
    scores = [judge.score_one(r.query.text, r.query.gold_answer, r.answer) for r in graded]
    return {
        "judge_correctness": sum(scores) / len(scores),
        "judge_cost_usd": judge.cost_usd,
        "judge_n": len(scores),
    }
