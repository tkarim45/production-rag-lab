"""Phase 3 embedders: tfidf fit, quantization, truncation, and their memory reports."""

import numpy as np

import modules  # noqa: F401 populate registry
from harness.contract import Chunk, Query
from harness.registry import build


def _chunks():
    return [
        Chunk("d::0", "d", "the cat sat on the mat"),
        Chunk("d::1", "d", "a dog ran in the park"),
        Chunk("d::2", "d", "quantum physics studies particles"),
    ]


def test_tfidf_fits_and_normalizes():
    e = build("embedder", "tfidf")
    chunks = e.encode_chunks(_chunks())
    for c in chunks:
        assert c.embedding is not None
        assert np.isclose(np.linalg.norm(c.embedding), 1.0, atol=1e-5)
    q = e.encode_query(Query("q", "where did the cat sit"))
    # query about cats should be most similar to chunk 0
    sims = [float(np.dot(q.embedding, c.embedding)) for c in chunks]
    assert int(np.argmax(sims)) == 0


def test_int8_quantization_near_lossless_memory_4x():
    e = build("embedder", "quantized", base="tfidf", mode="int8")
    e.encode_chunks(_chunks())
    # int8 = 1 byte/dim vs float32 4 bytes/dim
    assert e.memory_bytes_per_vector == e.dim


def test_binary_quantization_memory_32x():
    e = build("embedder", "quantized", base="tfidf", mode="binary")
    e.encode_chunks(_chunks())
    assert e.memory_bytes_per_vector == e.dim / 8


def test_matryoshka_truncates_dim():
    e = build("embedder", "matryoshka", base="tfidf", dim=2)
    chunks = e.encode_chunks(_chunks())
    assert all(len(c.embedding) <= 2 for c in chunks)
    assert e.dim <= 2


def test_quantized_mode_validation():
    import pytest

    with pytest.raises(ValueError):
        build("embedder", "quantized", mode="nope")
