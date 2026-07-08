"""YAML config → a built Pipeline.

A config names each stage's component (by registry name) plus its params, e.g.:

    dataset: builtin_mini
    retrieve_k: 10
    final_k: 5
    chunker:   {name: fixed, size: 256}
    embedder:  {name: hashing, dim: 512}
    index:     {name: flat}
    retriever: {name: dense}
    reranker:  null           # optional
    assembler: {name: concat}
    generator: {name: extractive_mock}

The loader instantiates each via the registry (so no imports leak into configs), wires the
retriever to the index, and returns (config_dict, Pipeline). Importing this module ensures
`modules` is imported so every component is registered first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import modules  # noqa: F401  (populates the registry)
from harness.pipeline import Pipeline
from harness.registry import build


def _component(cfg: dict[str, Any] | None, kind: str):
    if cfg is None:
        return None
    params = dict(cfg)
    name = params.pop("name")
    return build(kind, name, **params)


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    data.setdefault("dataset", "builtin_mini")
    data.setdefault("retrieve_k", 10)
    data.setdefault("final_k", 5)
    data.setdefault("reranker", None)
    data.setdefault("retriever", {"name": "dense"})
    data.setdefault("query_transformer", None)
    return data


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    index = _component(cfg["index"], "index")
    retriever = _component(cfg.get("retriever"), "retriever")
    # corpus binding happens in Pipeline.build (needs the chunks); here just bind the index
    # for the simple dense retriever so it works even if build() isn't reached in a test.
    if retriever is not None and hasattr(retriever, "bind") and not hasattr(retriever, "bind_corpus"):
        retriever.bind(index)

    pipeline = Pipeline(
        chunker=_component(cfg["chunker"], "chunker"),
        embedder=_component(cfg["embedder"], "embedder"),
        index=index,
        reranker=_component(cfg.get("reranker"), "reranker"),
        retriever=retriever,
        query_transformer=_component(cfg.get("query_transformer"), "query_transformer"),
        assembler=_component(cfg["assembler"], "assembler"),
        generator=_component(cfg["generator"], "generator"),
        retrieve_k=int(cfg["retrieve_k"]),
        final_k=int(cfg["final_k"]),
    )
    return pipeline


def load(path: str | Path) -> tuple[dict[str, Any], Pipeline]:
    cfg = load_config(path)
    return cfg, build_pipeline(cfg)
