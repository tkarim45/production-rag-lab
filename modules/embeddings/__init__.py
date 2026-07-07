"""Embeddings (Phase 3) — better encoders + the quantization/dim tradeoffs, deps-light.

Registers: `tfidf` (corpus-fit lexical embedder, beats hashing), and quantization/truncation
wrappers (`quantized`, `matryoshka`) that wrap any base embedder to study recall vs memory.
An on-disk embedding cache avoids recompute across sweeps. The optional
`sentence_transformer` embedder (real neural, MTEB-grade) registers from `modules.baseline`
when `.[embed]` is installed.
"""

from modules.embeddings import embedders  # noqa: F401
