"""Query transformers (Phase 6). `expand(query, retrieve_fn) -> list[Query]`."""

from __future__ import annotations

import os
import re
from collections import Counter

from harness.contract import Query
from harness.registry import register

_WORD = re.compile(r"[a-z0-9]+")
# tiny stoplist so PRF doesn't expand with function words
_STOP = set(
    "the a an and or of to in on for is are was were be by with as at it its this that from "
    "which who what when where how why not no do does did can could would should will".split()
)


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


@register("query_transformer", "prf")
class PRFExpander:
    """Pseudo-relevance feedback: assume the top-`fb_docs` retrieved are relevant, add their
    most frequent (non-stopword, non-query) terms to the query, retrieve again."""

    name = "prf"

    def __init__(self, fb_docs: int = 3, add_terms: int = 5):
        self.fb_docs, self.add_terms = fb_docs, add_terms

    def _expand_text(self, query: Query, retrieve_fn) -> str:
        seed = retrieve_fn(query, self.fb_docs)
        qterms = set(_terms(query.text))
        counts: Counter = Counter()
        for s in seed:
            for w in _terms(s.chunk.text):
                if w not in qterms:
                    counts[w] += 1
        extra = [w for w, _ in counts.most_common(self.add_terms)]
        return (query.text + " " + " ".join(extra)).strip() if extra else query.text

    def expand(self, query: Query, retrieve_fn) -> list[Query]:
        text = self._expand_text(query, retrieve_fn)
        return [Query(query_id=query.query_id, text=text, gold_answer=query.gold_answer,
                      relevant_chunk_ids=query.relevant_chunk_ids,
                      relevant_doc_ids=query.relevant_doc_ids, hop_type=query.hop_type)]


@register("query_transformer", "multiquery_prf")
class MultiQueryPRF(PRFExpander):
    """Return [original, PRF-expanded] so the pipeline RRF-fuses both retrievals — keeps the
    precision of the original while adding the recall of the expansion."""

    name = "multiquery_prf"

    def expand(self, query: Query, retrieve_fn) -> list[Query]:
        expanded = super().expand(query, retrieve_fn)[0]
        return [query, expanded]


def _register_llm() -> None:
    @register("query_transformer", "hyde")
    class HyDEExpander:  # pragma: no cover - needs creds
        """HyDE: ask the LLM for a hypothetical answer, retrieve with THAT (its vocabulary
        matches the target passages better than a short question)."""

        name = "hyde"

        def __init__(self, model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0", region: str | None = None):
            from anthropic import AnthropicBedrock

            region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._c = AnthropicBedrock(aws_region=region)
            self.model = model

        def expand(self, query: Query, retrieve_fn):
            msg = self._c.messages.create(
                model=self.model, max_tokens=120, temperature=0,
                messages=[{"role": "user", "content":
                           f"Write one concise factual sentence that would answer: {query.text}"}],
            )
            hypo = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            return [Query(query_id=query.query_id, text=f"{query.text} {hypo}",
                          gold_answer=query.gold_answer, relevant_chunk_ids=query.relevant_chunk_ids,
                          relevant_doc_ids=query.relevant_doc_ids, hop_type=query.hop_type)]


_register_llm()
