"""Faithfulness / groundedness + context metrics (Phase 11) — deps-free.

`faithfulness` here is the cheap, honest proxy: the fraction of the answer's content words
that actually appear in the retrieved context. It is NOT claim-level NLI entailment (that
needs a model) — it's a lexical *lower bound* on groundedness that runs on every query for
free. Its job is to catch the blatant case: an answer full of words that appear nowhere in
the context is ungrounded. Its blind spot: a paraphrase scores low despite being grounded,
and a fluent lie built from context words scores high. Documented, not hidden.
"""

from __future__ import annotations

import re

from harness.contract import PipelineResult

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an and or of to in on for is are was were be by with as at it its this that from "
    "which who what when where how why not no do does did can could would should will there "
    "their they them he she his her i you we our your".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def answer_groundedness(answer: str, context: str) -> float:
    """Fraction of the answer's content words present in the context. 1.0 = fully lexically
    grounded, 0.0 = nothing in the answer came from the context."""
    a = _content_words(answer)
    if not a:
        return 1.0          # empty/stopword-only answer asserts nothing → vacuously grounded
    c = _content_words(context)
    return len(a & c) / len(a)


def context_precision(result: PipelineResult) -> float:
    """Fraction of retrieved chunks that came from a relevant doc — noise in the window."""
    if not result.retrieved:
        return 0.0
    rel = result.query.relevant_doc_ids
    if not rel:
        return 0.0
    hits = sum(1 for s in result.retrieved if s.chunk.doc_id in rel)
    return hits / len(result.retrieved)


def context_recall(result: PipelineResult) -> float:
    """Fraction of the gold answer's content words covered by the assembled context —
    'could the model possibly have answered from what we gave it?'"""
    gold = result.query.gold_answer
    if not gold:
        return 0.0
    g = _content_words(gold)
    if not g:
        return 1.0
    c = _content_words(result.context)
    return len(g & c) / len(g)


def score_batch(results: list[PipelineResult]) -> dict[str, float]:
    n = len(results)
    if not n:
        return {}
    return {
        "groundedness": sum(answer_groundedness(r.answer, r.context) for r in results) / n,
        "context_precision": sum(context_precision(r) for r in results) / n,
        "context_recall": sum(context_recall(r) for r in results) / n,
    }
