"""Embedders.

Phase 0 default is a deterministic, dependency-free **hashing bag-of-words embedder** so the
harness runs and tests are reproducible with no model download. It is a real (if weak)
lexical embedding: token → hashed dim, L2-normalized, so cosine ≈ normalized term overlap.
That is enough for the naive baseline to retrieve the right chunk on the built-in set and
gives every later embedder (Phase 3: MiniLM/BGE, quantized, fine-tuned) a floor to beat.

The optional `sentence_transformer` embedder is registered only if the package is present.
"""

from __future__ import annotations

import re

import numpy as np

from harness.contract import Chunk, Query
from harness.registry import register

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@register("embedder", "hashing")
class HashingEmbedder:
    """Hashed bag-of-words → fixed-dim L2-normalized vector. Deterministic, key-free."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            # stable hash (Python's hash() is salted per-process) → md5-free, fast
            h = 2166136261
            for ch in tok:  # FNV-1a
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            c.embedding = self._embed(c.text)
        return chunks

    def encode_query(self, query: Query) -> Query:
        query.embedding = self._embed(query.text)
        return query


def _register_sentence_transformer() -> None:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        return  # optional; only available with `.[embed]`

    @register("embedder", "sentence_transformer")
    class SentenceTransformerEmbedder:  # pragma: no cover - needs the optional dep
        name = "sentence_transformer"

        def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(model)
            self.dim = self._m.get_sentence_embedding_dimension()

        def encode_chunks(self, chunks):
            embs = self._m.encode([c.text for c in chunks], normalize_embeddings=True)
            for c, e in zip(chunks, embs):
                c.embedding = np.asarray(e, dtype=np.float32)
            return chunks

        def encode_query(self, query):
            query.embedding = np.asarray(
                self._m.encode([query.text], normalize_embeddings=True)[0], dtype=np.float32
            )
            return query


_register_sentence_transformer()
