"""Contextual Retrieval (Phase 9) — chunker wrappers that prepend situating context.

The problem it fixes: a chunk like "Its revenue grew 3%" is unretrievable — "Its" refers to a
company named 20 chunks earlier. Anthropic's fix (Sept 2024) is to prepend a short blurb
situating the chunk in its document *before* embedding it. Reported retrieval-failure
reduction: 35% (contextual embeddings) → 49% (+ contextual BM25) → 67% (+ reranking).

`contextual` uses the structural context already available from ingestion metadata (title /
doc id) — deps-free and deterministic. `contextual_llm` generates the blurb with Claude, the
paper's actual method (optional; prompt-caching makes the per-chunk cost tolerable).
"""

from __future__ import annotations

import os

from harness.contract import Chunk, Document
from harness.registry import build, register


@register("chunker", "contextual")
class ContextualChunker:
    """Wrap a base chunker; prefix each chunk with its document's title/context so the
    embedding carries the situating information the chunk text alone lacks."""

    name = "contextual"

    def __init__(self, base: str = "fixed", **base_params):
        self._base = build("chunker", base, **base_params)

    def _context_for(self, doc_id: str, docs: list[Document]) -> str:
        for d in docs:
            if d.doc_id == doc_id:
                title = d.metadata.get("title") or d.metadata.get("title_path") or d.doc_id
                return str(title)
        return doc_id

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = self._base.run(docs)
        for c in chunks:
            ctx = self._context_for(c.doc_id, docs)
            c.metadata["context_prefix"] = ctx
            c.metadata["raw_text"] = c.text
            c.text = f"{ctx}. {c.text}"      # what gets embedded AND BM25-indexed
        return chunks


def _register_llm() -> None:
    @register("chunker", "contextual_llm")
    class ContextualLLMChunker:  # pragma: no cover - needs creds
        """The paper's method: Claude writes a one-sentence blurb situating each chunk in its
        document, prepended before embedding."""

        name = "contextual_llm"

        def __init__(self, base: str = "fixed", model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                     region: str | None = None, **base_params):
            from anthropic import AnthropicBedrock

            self._base = build("chunker", base, **base_params)
            region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            self._c = AnthropicBedrock(aws_region=region)
            self.model = model

        def _blurb(self, doc_text: str, chunk_text: str) -> str:
            msg = self._c.messages.create(
                model=self.model, max_tokens=60, temperature=0,
                system="Give a short one-sentence context situating the chunk within the document. "
                       "No preamble, just the sentence.",
                messages=[{"role": "user", "content":
                           f"<document>{doc_text[:4000]}</document>\n<chunk>{chunk_text}</chunk>"}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()

        def run(self, docs):
            by_id = {d.doc_id: d.text for d in docs}
            chunks = self._base.run(docs)
            for c in chunks:
                ctx = self._blurb(by_id.get(c.doc_id, ""), c.text)
                c.metadata["context_prefix"] = ctx
                c.metadata["raw_text"] = c.text
                c.text = f"{ctx} {c.text}"
            return chunks


_register_llm()
