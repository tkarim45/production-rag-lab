"""Context assembly (Phase 8).

Turning a ranked chunk list into the prompt is its own decision layer: which chunks, in what
order, deduped how, under what token budget, with citations preserved. Registers `reorder`
(lost-in-the-middle mitigation), `dedup`, `budget` (token-capped), and `parent` (small-to-big
expansion). Phase 0's `concat` is the naive baseline.
"""

from modules.context import assemblers  # noqa: F401
