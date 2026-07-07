"""Generators.

Two implementations:
- `extractive_mock` (default): deterministic, key-free. Returns the sentence from the
  assembled context with the highest lexical overlap with the query. No LLM, so the harness
  and CI run offline and the retrieval/latency metrics are pure. It gives token-F1 a real
  (non-zero, non-perfect) signal so cost-per-correct-answer has a denominator.
- `claude` (optional): real Claude on AWS Bedrock, temperature 0, grounded+cited prompt.
  Used once creds are present; reports real tokens + cost. Model id and price come from
  config so the leaderboard cost column is real.
"""

from __future__ import annotations

import os
import re
from typing import Any

from harness.contract import Query
from harness.registry import register

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@register("generator", "extractive_mock")
class ExtractiveMockGenerator:
    """Pick the context sentence with max token overlap with the query. Key-free."""

    name = "extractive_mock"

    def generate(self, query: Query, context: str) -> dict[str, Any]:
        # strip the "[n] (chunk_id)" citation prefixes before sentence splitting
        clean = re.sub(r"\[\d+\]\s*\([^)]*\)\s*", "", context)
        sentences = [s.strip() for s in _SENT.split(clean) if s.strip()]
        if not sentences:
            return {"answer": "", "tokens": {"in": 0, "out": 0}, "cost_usd": 0.0}
        q = _tok(query.text)
        best = max(sentences, key=lambda s: len(_tok(s) & q))
        return {
            "answer": best,
            "tokens": {"in": len(clean.split()), "out": len(best.split())},
            "cost_usd": 0.0,
        }


def _register_claude() -> None:
    @register("generator", "claude")
    class ClaudeGenerator:  # pragma: no cover - needs creds + optional dep
        """Real Claude on AWS Bedrock, temp 0, grounded + cited. Reports real tokens/cost."""

        name = "claude"

        def __init__(
            self,
            model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region: str | None = None,
            max_tokens: int = 512,
            price_in_per_mtok: float = 1.00,
            price_out_per_mtok: float = 5.00,
        ):
            from anthropic import AnthropicBedrock

            region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._client = AnthropicBedrock(aws_region=region)
            self.model = model
            self.max_tokens = max_tokens
            self.price_in = price_in_per_mtok
            self.price_out = price_out_per_mtok

        def generate(self, query: Query, context: str) -> dict[str, Any]:
            system = (
                "You are a retrieval-grounded assistant. Answer ONLY from the numbered "
                "context. Cite sources as [n]. If the context does not contain the answer, "
                "say you don't know. Be concise."
            )
            user = f"Context:\n{context}\n\nQuestion: {query.text}\nAnswer:"
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            answer = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
            cost = tin / 1e6 * self.price_in + tout / 1e6 * self.price_out
            return {"answer": answer.strip(), "tokens": {"in": tin, "out": tout}, "cost_usd": cost}


_register_claude()
