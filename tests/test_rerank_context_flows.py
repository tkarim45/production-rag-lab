"""Phase 7 rerankers, Phase 8 assemblers, Phase 9 contextual retrieval."""

import numpy as np

import modules  # noqa: F401
from harness.contract import Chunk, Document, Query, Scored
from harness.registry import build


# ── Phase 7: reranking ─────────────────────────────────────────────────────────

def test_lexical_reranker_promotes_the_lexical_match():
    cands = [
        Scored(Chunk("a::0", "a", "unrelated text about weather patterns"), 0.9),  # ranked 1st
        Scored(Chunk("b::0", "b", "the capital of france is paris"), 0.1),         # ranked last
    ]
    rr = build("reranker", "lexical")
    out = rr.rerank(Query("q", "what is the capital of france"), cands, k=2)
    assert out[0].chunk.doc_id == "b"     # reranker fixes the first stage's mistake


def test_lexical_reranker_respects_k_and_empty():
    rr = build("reranker", "lexical")
    assert rr.rerank(Query("q", "x"), [], k=3) == []
    cands = [Scored(Chunk(f"{i}::0", str(i), f"doc {i} text"), 0.5) for i in range(5)]
    assert len(rr.rerank(Query("q", "doc"), cands, k=2)) == 2


# ── Phase 8: context assembly ──────────────────────────────────────────────────

def _scored(n=4, dim=4):
    vecs = np.eye(max(n, dim), dtype=np.float32)
    return [Scored(Chunk(f"d{i}::0", f"d{i}", f"chunk {i} words here", embedding=vecs[i]), 1.0 - i * 0.1)
            for i in range(n)]


def test_reorder_puts_best_at_the_edges():
    out = build("assembler", "reorder").assemble(Query("q", "x"), _scored(4))
    # best (chunk 0) first, second-best (chunk 1) last
    assert out.index("chunk 0") < out.index("chunk 2")
    assert out.rstrip().endswith("chunk 1 words here")


def test_dedup_drops_near_duplicates():
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    scored = [
        Scored(Chunk("a::0", "a", "identical content", embedding=v.copy()), 1.0),
        Scored(Chunk("b::0", "b", "identical content twin", embedding=v.copy()), 0.9),  # cosine 1.0
        Scored(Chunk("c::0", "c", "different", embedding=np.array([0, 1, 0, 0], dtype=np.float32)), 0.8),
    ]
    out = build("assembler", "dedup", threshold=0.9).assemble(Query("q", "x"), scored)
    assert "identical content twin" not in out
    assert "different" in out


def test_budget_caps_words():
    scored = [Scored(Chunk(f"d{i}::0", f"d{i}", " ".join(["word"] * 50)), 1.0) for i in range(5)]
    out = build("assembler", "budget", max_words=120).assemble(Query("q", "x"), scored)
    # 2 chunks (100 words) fit, the 3rd (150) would bust the budget
    assert out.count("[") == 2


def test_parent_expands_to_parent_text():
    c = Chunk("a::0", "a", "small child", metadata={"parent_text": "the full parent paragraph"})
    out = build("assembler", "parent").assemble(Query("q", "x"), [Scored(c, 1.0)])
    assert "the full parent paragraph" in out


# ── Phase 9: contextual retrieval ──────────────────────────────────────────────

def test_contextual_chunker_prefixes_title_and_keeps_raw():
    docs = [Document("t01", "Some body text about revenue growth.", {"title": "NovaChip 10-K"})]
    chunks = build("chunker", "contextual", base="fixed", size=60).run(docs)
    assert chunks[0].text.startswith("NovaChip 10-K.")
    assert chunks[0].metadata["raw_text"] == "Some body text about revenue growth."
    assert chunks[0].metadata["context_prefix"] == "NovaChip 10-K"


def test_contextual_chunker_preserves_chunk_ids():
    docs = [Document("d1", " ".join(["word"] * 200), {"title": "T"})]
    chunks = build("chunker", "contextual", base="fixed", size=60, overlap=10).run(docs)
    assert [c.chunk_id for c in chunks] == [f"d1::{i}" for i in range(len(chunks))]
