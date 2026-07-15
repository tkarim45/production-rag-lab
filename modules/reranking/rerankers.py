"""Reranker implementations (Phase 7). `rerank(query, scored, k) -> list[Scored]`.

All are *pointwise*: score each (query, candidate) pair independently, then sort. That is
the cross-encoder pattern — and the reason cost is O(candidates) forward passes, unlike the
bi-encoder's O(1) query encode.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

from harness.contract import Query, Scored
from harness.registry import register

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on for is are was were be by with as at it its this that".split())


def _tok(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@register("reranker", "lexical")
class LexicalReranker:
    """Deps-free pointwise reranker: BM25-style scoring of the query against each candidate
    *in isolation* (IDF computed over the candidate set). Not a neural cross-encoder, but the
    same shape — it re-scores pairs the first stage ranked by a cheaper signal, and it's a
    real, honest lift when the first stage is lexically weak."""

    name = "lexical"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def rerank(self, query: Query, scored: list[Scored], k: int) -> list[Scored]:
        if not scored:
            return []
        docs = [_tok(s.chunk.text) for s in scored]
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n if n else 0.0
        df: Counter = Counter()
        for d in docs:
            for w in set(d):
                df[w] += 1
        q = _tok(query.text)
        out: list[Scored] = []
        for s, d in zip(scored, docs):
            tf = Counter(d)
            score = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * len(d) / (avgdl or 1))
                score += idf * (f * (self.k1 + 1)) / denom
            out.append(Scored(chunk=s.chunk, score=score))
        out.sort(key=lambda s: -s.score)
        return out[:k]


def _register_cross_encoder() -> None:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except Exception:
        return

    @register("reranker", "cross_encoder")
    class CrossEncoderReranker:  # pragma: no cover - optional heavy dep
        """The real thing: jointly encode [query, passage] → relevance score."""

        name = "cross_encoder"

        def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            from sentence_transformers import CrossEncoder

            self._m = CrossEncoder(model)

        def rerank(self, query, scored, k):
            if not scored:
                return []
            pairs = [(query.text, s.chunk.text) for s in scored]
            scores = self._m.predict(pairs)
            out = [Scored(chunk=s.chunk, score=float(sc)) for s, sc in zip(scored, scores)]
            out.sort(key=lambda s: -s.score)
            return out[:k]


def _register_llm() -> None:
    @register("reranker", "llm")
    class LLMReranker:  # pragma: no cover - needs creds
        """Pointwise LLM reranker: ask the model to rate each passage's relevance 0-10."""

        name = "llm"

        def __init__(self, model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                     region: str | None = None):
            from anthropic import AnthropicBedrock

            region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._c = AnthropicBedrock(aws_region=region)
            self.model = model

        def _score(self, q: str, passage: str) -> float:
            msg = self._c.messages.create(
                model=self.model, max_tokens=8, temperature=0,
                system="Rate how well the passage answers the question. Reply with ONLY an integer 0-10.",
                messages=[{"role": "user", "content": f"Question: {q}\n\nPassage: {passage}\n\nScore:"}],
            )
            txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            m = re.search(r"\d+", txt)
            return float(m.group()) if m else 0.0

        def rerank(self, query, scored, k):
            out = [Scored(chunk=s.chunk, score=self._score(query.text, s.chunk.text)) for s in scored]
            out.sort(key=lambda s: -s.score)
            return out[:k]


_register_cross_encoder()
_register_llm()
