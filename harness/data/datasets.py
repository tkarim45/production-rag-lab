"""Dataset loaders → (docs, queries).

Phase 0 ships one small, fully-labeled built-in dataset (`builtin_mini`) so the harness
runs end-to-end offline with real qrels + gold answers and the tests are deterministic.
Real benchmark loaders (BEIR subsets with qrels, HotpotQA/SQuAD for QA) are declared here
and become available once `pip install -e ".[data]"` is done and the corpora are fetched —
they return the SAME (docs, queries) shape, so nothing downstream changes.

A "dataset" is: list[Document] + list[Query] where each Query carries its gold answer and
the set of relevant chunk ids. NOTE the built-in corpus is authored so that each document
is a single chunk under the naive fixed-size chunker with `chunk_id == f"{doc_id}::0"`,
which is what the qrels reference. Larger real corpora label at the passage level.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.contract import Document, Query

_BUILTIN = Path(__file__).parent / "builtin"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_pair(corpus_file: str, query_file: str) -> tuple[list[Document], list[Query]]:
    corpus = _read_jsonl(_BUILTIN / corpus_file)
    qa = _read_jsonl(_BUILTIN / query_file)
    docs = [Document(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {})) for d in corpus]
    queries = [
        Query(
            query_id=q["query_id"],
            text=q["text"],
            gold_answer=q.get("gold_answer"),
            relevant_chunk_ids=set(q.get("relevant_chunk_ids", [])),
            relevant_doc_ids=set(q.get("relevant_doc_ids", [])),
            hop_type=q.get("hop_type", "single"),
        )
        for q in qa
    ]
    return docs, queries


def _load_builtin_mini() -> tuple[list[Document], list[Query]]:
    # single-chunk docs, chunk-level qrels — fast smoke set
    return _load_pair("corpus.jsonl", "queries.jsonl")


def _load_builtin_docs() -> tuple[list[Document], list[Query]]:
    # multi-paragraph docs + distractors, doc-level qrels — used from Phase 2 onward so
    # chunking / retrieval / rerank choices actually differentiate.
    return _load_pair("docs_corpus.jsonl", "docs_queries.jsonl")


def _load_beir(_name: str) -> tuple[list[Document], list[Query]]:  # pragma: no cover - Phase 0 stub
    raise NotImplementedError(
        "BEIR loader lands with Phase 0's real-data step. Install `.[data]`, fetch the "
        "subset (fiqa/scifact/nfcorpus), and map qrels → relevant_chunk_ids here. Returns "
        "the same (docs, queries) shape as builtin_mini."
    )


_LOADERS = {
    "builtin_mini": _load_builtin_mini,
    "builtin_docs": _load_builtin_docs,
    "beir_fiqa": lambda: _load_beir("fiqa"),
    "beir_scifact": lambda: _load_beir("scifact"),
    "beir_nfcorpus": lambda: _load_beir("nfcorpus"),
}


def available_datasets() -> list[str]:
    return sorted(_LOADERS)


def load_dataset(name: str) -> tuple[list[Document], list[Query]]:
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {available_datasets()}") from None
    return loader()
