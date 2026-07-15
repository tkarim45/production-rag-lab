"""Generation (Phase 10): one Claude generator, parameterized by prompt style + decoding.

Prompt styles (the lever the phase measures):
- `bare`       — no grounding instruction. The control.
- `grounded`   — answer only from context.
- `cite_forced`— answer only from context AND cite [n]; refuse if uncited.
- `abstain`    — grounded + explicit permission to say "I don't know" when context lacks it.

Decoding: temperature / top_p (temp 0 is the eval default — the honest reason is
reproducibility, not accuracy per se).
"""

from __future__ import annotations

import os
import re
from typing import Any

from harness.contract import Query
from harness.registry import register

_SYSTEMS = {
    "bare": "You are a helpful assistant. Answer the question. Be concise.",
    "grounded": (
        "Answer ONLY using the numbered context provided. Do not use outside knowledge. "
        "Be concise."
    ),
    "cite_forced": (
        "Answer ONLY using the numbered context. Every claim must cite its source as [n]. "
        "If the context does not support an answer, reply exactly: I don't know. Be concise."
    ),
    "abstain": (
        "Answer ONLY using the numbered context. If the context does not contain the answer, "
        "reply exactly: I don't know. Do not guess. Be concise."
    ),
}


def _register() -> None:
    @register("generator", "claude_prompted")
    class ClaudePromptedGenerator:  # pragma: no cover - needs creds
        name = "claude_prompted"

        def __init__(
            self,
            style: str = "grounded",
            temperature: float = 0.0,
            top_p: float | None = None,
            model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region: str | None = None,
            max_tokens: int = 256,
            price_in_per_mtok: float = 1.00,
            price_out_per_mtok: float = 5.00,
        ):
            from anthropic import AnthropicBedrock

            if style not in _SYSTEMS:
                raise ValueError(f"style must be one of {sorted(_SYSTEMS)}")
            region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._c = AnthropicBedrock(aws_region=region)
            self.style, self.temperature, self.top_p = style, temperature, top_p
            self.model, self.max_tokens = model, max_tokens
            self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok

        def generate(self, query: Query, context: str) -> dict[str, Any]:
            kwargs: dict[str, Any] = dict(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                system=_SYSTEMS[self.style],
                messages=[{"role": "user", "content":
                           f"Context:\n{context}\n\nQuestion: {query.text}\nAnswer:"}],
            )
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            msg = self._c.messages.create(**kwargs)
            answer = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
            cost = tin / 1e6 * self.price_in + tout / 1e6 * self.price_out
            return {
                "answer": answer,
                "tokens": {"in": tin, "out": tout},
                "cost_usd": cost,
                # Phase 10/11 signals, computed here so scoring can read them off the answer
                "has_citation": bool(re.search(r"\[\d+\]", answer)),
                "abstained": answer.lower().startswith("i don't know"),
            }


_register()
