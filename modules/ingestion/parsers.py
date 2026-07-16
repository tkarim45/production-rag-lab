"""Format parsers: raw file → plain text. Deps-free for text/markdown/HTML/CSV.

PDF/DOCX/OCR need optional deps (pymupdf / python-docx / pytesseract) and register only if
importable — the ingestor skips a format cleanly if its parser isn't installed, so Phase 1
runs on the stdlib. Each parser returns (text, parse_meta) where parse_meta records fidelity
signals (e.g. tables found, chars extracted) for the quality report.
"""

from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

# extension → parser(path) -> (text, meta)
_PARSERS: dict[str, Callable[[Path], tuple[str, dict[str, Any]]]] = {}


def parser(*exts: str):
    def deco(fn):
        for e in exts:
            _PARSERS[e.lower()] = fn
        return fn

    return deco


def supported_extensions() -> list[str]:
    return sorted(_PARSERS)


def parse(path: Path) -> tuple[str, dict[str, Any]]:
    ext = path.suffix.lower()
    if ext not in _PARSERS:
        raise ValueError(f"no parser for {ext!r} (supported: {supported_extensions()})")
    return _PARSERS[ext](path)


@parser(".txt")
def _parse_txt(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, {"format": "txt", "chars": len(text)}


@parser(".md", ".markdown")
def _parse_md(path: Path) -> tuple[str, dict[str, Any]]:
    import re

    raw = path.read_text(encoding="utf-8", errors="replace")
    # strip markdown syntax to plain text, keep the words
    text = re.sub(r"`{1,3}[^`]*`{1,3}", " ", raw)          # code
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links/images → link text
    text = re.sub(r"[#>*_\-]{1,}", " ", text)               # heading/list/emphasis marks
    return text, {"format": "markdown", "chars": len(text)}


class _HTMLTextExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "footer", "head"}

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


@parser(".html", ".htm")
def _parse_html(path: Path) -> tuple[str, dict[str, Any]]:
    ex = _HTMLTextExtractor()
    ex.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = ex.text()
    return text, {"format": "html", "chars": len(text), "boilerplate_stripped": True}


@parser(".csv")
def _parse_csv(path: Path) -> tuple[str, dict[str, Any]]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))))
    if not rows:
        return "", {"format": "csv", "rows": 0}
    header, *body = rows
    # linearize each row as "col: val" sentences so tables are retrievable prose
    lines = []
    for r in body:
        lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, r)))
    return "\n".join(lines), {"format": "csv", "rows": len(body), "cols": len(header), "table": True}


def _register_optional() -> None:
    try:
        import fitz  # noqa: F401  (pymupdf)

        @parser(".pdf")
        def _parse_pdf(path: Path):  # pragma: no cover - optional dep
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            return text, {"format": "pdf", "pages": doc.page_count, "chars": len(text)}
    except Exception:
        pass

    try:
        import docx  # noqa: F401  (python-docx)

        @parser(".docx")
        def _parse_docx(path: Path):  # pragma: no cover - optional dep
            d = docx.Document(str(path))
            text = "\n".join(p.text for p in d.paragraphs)
            return text, {"format": "docx", "chars": len(text)}
    except Exception:
        pass


_register_optional()
