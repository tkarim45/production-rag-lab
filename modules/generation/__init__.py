"""Generation (Phase 10).

The last mile: how you prompt and decode decides whether good retrieval becomes a good,
grounded, citable answer — or a confident hallucination. Registers `claude_prompted`, a
Claude/Bedrock generator parameterized by **prompt style** (bare / grounded / cite_forced /
abstain) and **decoding params** (temperature, top_p), so the phase can sweep them on one
harness and measure the difference. Phase 0's `extractive_mock` remains the key-free floor.
"""

from modules.generation import generators  # noqa: F401
