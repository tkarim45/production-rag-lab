"""Component registry — the mechanism that makes stages swappable from a config file.

Every stage implementation registers itself under a `(kind, name)` key with a factory.
A YAML config then names a component by kind+name and passes params; the config loader
looks it up here and instantiates it. This is what lets `configs/naive.yaml` say
`chunker: {name: fixed, size: 256}` and get the right class with zero imports in the config.

Usage in a module:

    from harness.registry import register

    @register("chunker", "fixed")
    class FixedSizeChunker:
        def __init__(self, size=256, overlap=0): ...
        def run(self, docs): ...
"""

from __future__ import annotations

from typing import Any, Callable

# kind → name → factory(**params)
_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}

KINDS = ("chunker", "embedder", "index", "retriever", "reranker", "assembler", "generator")


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if kind not in KINDS:
        raise ValueError(f"unknown stage kind {kind!r}; expected one of {KINDS}")

    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket:
            raise ValueError(f"{kind} {name!r} already registered")
        bucket[name] = factory
        return factory

    return deco


def build(kind: str, name: str, **params: Any) -> Any:
    try:
        factory = _REGISTRY[kind][name]
    except KeyError:
        available = sorted(_REGISTRY.get(kind, {}))
        raise KeyError(
            f"no {kind} registered as {name!r}. Available: {available}. "
            f"Did you import the module that registers it?"
        ) from None
    return factory(**params)


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def all_components() -> dict[str, list[str]]:
    return {kind: available(kind) for kind in KINDS}
