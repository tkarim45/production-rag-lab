"""Ingestor: a directory of raw files → clean, deduped, metadata-rich Documents + a report.

Pipeline: for each file → parse (by extension) → clean/normalize → build a Document with
enriched metadata (source path, format, language, page/table signals). Then exact + near-dup
dedup across the whole set. Emits an IngestionReport (parse fidelity, dedup rate, counts) —
the Phase 1 deliverable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.contract import Document
from modules.ingestion import clean as C
from modules.ingestion import dedup as D
from modules.ingestion import parsers as P


@dataclass
class IngestionReport:
    n_files_seen: int = 0
    n_parsed: int = 0
    n_skipped_unsupported: list[str] = field(default_factory=list)
    formats: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    n_after_dedup: int = 0
    exact_dropped: list = field(default_factory=list)
    near_dropped: list = field(default_factory=list)
    total_chars_raw: int = 0
    total_chars_clean: int = 0

    @property
    def dedup_rate(self) -> float:
        dropped = len(self.exact_dropped) + len(self.near_dropped)
        return dropped / self.n_parsed if self.n_parsed else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dedup_rate"] = round(self.dedup_rate, 4)
        return d


def ingest_dir(
    root: str | Path, dedup_threshold: float = 0.8
) -> tuple[list[Document], IngestionReport]:
    root = Path(root)
    report = IngestionReport()
    supported = set(P.supported_extensions())
    raw_docs: list[Document] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        report.n_files_seen += 1
        if path.suffix.lower() not in supported:
            report.n_skipped_unsupported.append(path.name)
            continue
        text, pmeta = P.parse(path)
        clean_text, cmeta = C.clean(text)
        if not clean_text:
            continue
        report.n_parsed += 1
        report.total_chars_raw += pmeta.get("chars", len(text))
        report.total_chars_clean += len(clean_text)
        fmt = pmeta.get("format", "unknown")
        lang = cmeta.get("language", "unknown")
        report.formats[fmt] = report.formats.get(fmt, 0) + 1
        report.languages[lang] = report.languages.get(lang, 0) + 1

        raw_docs.append(
            Document(
                doc_id=path.stem,
                text=clean_text,
                metadata={
                    "source": str(path.relative_to(root)),
                    "title_path": path.stem.replace("_", " ").title(),
                    **{k: v for k, v in pmeta.items() if k != "chars"},
                    "language": lang,
                },
            )
        )

    kept, dreport = D.dedup(raw_docs, threshold=dedup_threshold)
    report.n_after_dedup = len(kept)
    report.exact_dropped = dreport.exact_dropped
    report.near_dropped = dreport.near_dropped
    return kept, report
