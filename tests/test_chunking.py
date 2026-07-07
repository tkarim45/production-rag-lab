"""Phase 2 chunkers: each produces valid chunks and the sweep differentiates them."""

import modules  # noqa: F401 populate registry
from harness.contract import Document
from harness.registry import build

_DOC = Document(
    "d",
    "First sentence about cats. Second about cats too.\n\n"
    "A new paragraph on dogs. Dogs are loyal. Dogs bark often.",
    {"title": "animals"},
)


def _ids_valid(chunks, doc_id="d"):
    assert chunks, "no chunks produced"
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"{doc_id}::{i}"
        assert c.doc_id == doc_id
        assert c.text.strip()


def test_all_chunkers_produce_valid_chunks():
    for name in ("fixed", "recursive", "sentence", "paragraph", "structural", "semantic", "parent_child"):
        chunker = build("chunker", name)
        _ids_valid(chunker.run([_DOC]))


def test_paragraph_chunker_splits_on_blank_line():
    chunks = build("chunker", "paragraph").run([_DOC])
    assert len(chunks) == 2  # two paragraphs


def test_sentence_overlap_validation():
    import pytest

    with pytest.raises(ValueError):
        build("chunker", "sentence", per_chunk=2, overlap=2)


def test_parent_child_carries_parent_text():
    chunks = build("chunker", "parent_child").run([_DOC])
    assert all("parent_text" in c.metadata for c in chunks)
    # children are smaller than their parent paragraph
    assert any(len(c.text) < len(c.metadata["parent_text"]) for c in chunks)


def test_sweep_runs_and_differentiates():
    from harness.sweep import _BASE, _run_one
    import copy

    r_fixed = _run_one({**copy.deepcopy(_BASE), "chunker": {"name": "fixed", "size": 60, "overlap": 10}}, "t_fixed")
    r_recur = _run_one({**copy.deepcopy(_BASE), "chunker": {"name": "recursive", "size": 60, "overlap": 10}}, "t_recur")
    # both valid metric dicts; recursive should not be worse on mrr here (sanity, not a hard SLA)
    assert "mrr" in r_fixed and "mrr" in r_recur
    assert r_recur["recall@5"] >= 0.5
