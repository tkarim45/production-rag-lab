"""Registry + config loader wiring."""

import pytest

import modules  # noqa: F401  populates registry
from harness import config as cfgmod
from harness.registry import all_components, build


def test_baseline_components_registered():
    comps = all_components()
    assert "fixed" in comps["chunker"]
    assert "hashing" in comps["embedder"]
    assert "flat" in comps["index"]
    assert "dense" in comps["retriever"]
    assert "concat" in comps["assembler"]
    assert "extractive_mock" in comps["generator"]


def test_build_unknown_raises_helpful():
    with pytest.raises(KeyError, match="no chunker registered"):
        build("chunker", "does_not_exist")


def test_load_naive_config_builds_pipeline():
    cfg, pipeline = cfgmod.load("configs/naive.yaml")
    assert cfg["dataset"] == "builtin_mini"
    assert pipeline.chunker.name == "fixed"
    assert pipeline.embedder.name == "hashing"
    assert pipeline.reranker is None
    # dense retriever got the index bound
    assert pipeline.index is not None


def test_fixed_chunker_validates_overlap():
    from modules.baseline.chunkers import FixedSizeChunker

    with pytest.raises(ValueError):
        FixedSizeChunker(size=10, overlap=10)
