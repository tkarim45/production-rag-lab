"""Cleaning & normalization: whitespace, boilerplate, encoding, language heuristic.

Deliberately conservative — normalize whitespace and strip obvious boilerplate lines
(copyright, "all rights reserved") without touching content words, since aggressive
cleaning silently drops answerable text. Language detection is a cheap stopword heuristic
(no dep); Phase 1's optional `.[data]` can swap in a real detector.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_BOILERPLATE = re.compile(
    r"^\s*(©|\(c\)|copyright|all rights reserved|cookie|subscribe|advertisement)\b",
    re.IGNORECASE,
)

# tiny stopword sets for a language guess (enough to flag non-English in the report)
_STOP = {
    "en": {"the", "and", "of", "to", "in", "is", "it", "was", "on", "for"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "las", "un", "una"},
    "fr": {"le", "la", "de", "et", "les", "des", "un", "une", "dans", "est"},
}


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def strip_boilerplate(text: str) -> str:
    kept = [ln for ln in text.splitlines() if not _BOILERPLATE.match(ln)]
    return "\n".join(kept)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    if not words:
        return "unknown"
    wset = set(words)
    scores = {lang: len(wset & sw) for lang, sw in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def clean(text: str) -> tuple[str, dict]:
    """Return (clean_text, meta). meta carries language + a cleaning fidelity signal."""
    stripped = strip_boilerplate(text)
    normalized = normalize_whitespace(stripped)
    lang = detect_language(normalized)
    return normalized, {
        "language": lang,
        "chars_before": len(text),
        "chars_after": len(normalized),
    }
