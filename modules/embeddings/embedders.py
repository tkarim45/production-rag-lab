"""Embedder implementations + quantization/truncation wrappers (Phase 3).

- `tfidf`: fit IDF over the corpus during encode_chunks; encode queries with the fitted
  vocabulary. A real (stronger) lexical embedder — the floor the neural embedders must beat.
- `quantized`: wrap any base embedder and quantize its output to int8 or binary (1 bit/dim),
  measuring the recall vs memory/speed tradeoff. Search still uses float cosine on the
  dequantized vector so the metric reflects information lost, not index mechanics.
- `matryoshka`: truncate any base embedder to the first `dim` components (+ renormalize) to
  study the dimensionality/quality tradeoff.

`memory_bytes_per_vector` is reported so the leaderboard can weigh recall against footprint.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from harness.contract import Chunk, Query
from harness.registry import build, register

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@register("embedder", "tfidf")
class TfidfEmbedder:
    """Corpus-fit TF-IDF vectors, L2-normalized. Fits during encode_chunks."""

    name = "tfidf"

    def __init__(self, max_features: int = 4096):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.dim = 0

    def _fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            for w in set(_tok(t)):
                df[w] += 1
        # keep the most frequent terms up to max_features
        vocab_terms = [w for w, _ in df.most_common(self.max_features)]
        self.vocab = {w: i for i, w in enumerate(vocab_terms)}
        self.dim = len(self.vocab)
        n = len(texts)
        idf = np.zeros(self.dim, dtype=np.float32)
        for w, i in self.vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[w])) + 1.0
        self.idf = idf

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if self.dim == 0:
            return v
        tf = Counter(_tok(text))
        for w, c in tf.items():
            i = self.vocab.get(w)
            if i is not None:
                v[i] = c * self.idf[i]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        self._fit([c.text for c in chunks])
        for c in chunks:
            c.embedding = self._vec(c.text)
        return chunks

    def encode_query(self, query: Query) -> Query:
        query.embedding = self._vec(query.text)
        return query


class _Wrapper:
    """Shared plumbing for embedders that wrap a base embedder."""

    def __init__(self, base: str = "tfidf", **base_params):
        self.base = build("embedder", base, **base_params)

    def encode_query(self, query: Query) -> Query:
        query = self.base.encode_query(query)
        query.embedding = self._transform(query.embedding)
        return query


@register("embedder", "quantized")
class QuantizedEmbedder(_Wrapper):
    """Quantize base embeddings to int8 or binary; dequantize for float cosine search."""

    name = "quantized"

    def __init__(self, base: str = "tfidf", mode: str = "int8", **base_params):
        super().__init__(base, **base_params)
        if mode not in ("int8", "binary"):
            raise ValueError("mode must be 'int8' or 'binary'")
        self.mode = mode

    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def memory_bytes_per_vector(self) -> float:
        if self.mode == "int8":
            return self.base.dim  # 1 byte/dim
        return self.base.dim / 8  # 1 bit/dim

    def _transform(self, v: np.ndarray) -> np.ndarray:
        if v is None:
            return v
        if self.mode == "int8":
            scale = np.abs(v).max() or 1.0
            q = np.round(v / scale * 127).astype(np.int8)
            deq = q.astype(np.float32) * scale / 127
        else:  # binary: sign bit, dequantize to ±1
            deq = np.where(v >= 0, 1.0, -1.0).astype(np.float32)
        n = np.linalg.norm(deq)
        return deq / n if n > 0 else deq

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        chunks = self.base.encode_chunks(chunks)
        for c in chunks:
            c.embedding = self._transform(c.embedding)
        return chunks


@register("embedder", "matryoshka")
class MatryoshkaEmbedder(_Wrapper):
    """Truncate base embeddings to the first `dim` components and renormalize."""

    name = "matryoshka"

    def __init__(self, base: str = "tfidf", dim: int = 128, **base_params):
        super().__init__(base, **base_params)
        self._target = dim

    @property
    def dim(self) -> int:
        return min(self._target, self.base.dim)

    @property
    def memory_bytes_per_vector(self) -> float:
        return self.dim * 4  # float32

    def _transform(self, v: np.ndarray) -> np.ndarray:
        if v is None:
            return v
        t = v[: self._target]
        n = np.linalg.norm(t)
        return (t / n if n > 0 else t).astype(np.float32)

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        chunks = self.base.encode_chunks(chunks)
        for c in chunks:
            c.embedding = self._transform(c.embedding)
        return chunks
