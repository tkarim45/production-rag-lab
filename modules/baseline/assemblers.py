"""Context assemblers. Phase 0 ships numbered concatenation with citation tags; Phase 8
adds dedup/cluster, lost-in-the-middle reordering, compression, and token budgeting."""

from __future__ import annotations

from harness.contract import Query, Scored
from harness.registry import register


@register("assembler", "concat")
class ConcatAssembler:
    """Concatenate retrieved chunks as numbered, citable context blocks.

    Each block is tagged `[n] (chunk_id)` so the generator can cite and downstream citation
    metrics (Phase 11) can check the cited chunk actually supports the claim.
    """

    name = "concat"

    def __init__(self, max_chars: int | None = None):
        self.max_chars = max_chars

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        blocks = []
        for i, s in enumerate(scored, start=1):
            blocks.append(f"[{i}] ({s.chunk.chunk_id}) {s.chunk.text}")
        context = "\n\n".join(blocks)
        if self.max_chars is not None and len(context) > self.max_chars:
            context = context[: self.max_chars]
        return context
