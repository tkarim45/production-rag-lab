"""PII detection & redaction (Phase 14) — at ingest and at output.

Two placements, two different threats:
- **At ingest** (`redact_documents`): the PII never enters the index, so it cannot be
  retrieved, cannot be embedded, cannot be cached, and cannot be leaked by a future bug.
  Irreversible, and irreversible is the point.
- **At output** (`redact_text`): the last line of defense for PII that reached the context
  anyway — from a document ingested before the policy existed, or from the user's own turn.

The interesting engineering is in credit cards. A regex for "13-19 digits" matches every
order number, invoice id, and account reference in the corpus, and a redactor with that
false-positive rate gets switched off within a week. **Luhn** (the checksum every card
carries) is the cheap discriminator: it rejects ~90% of random digit strings for free.

The honest limit — measured in the bench, not hand-waved — is that Luhn is a *checksum, not
a semantic classifier*. IMEIs use the same checksum. A Luhn-valid IMEI is indistinguishable
from a card by arithmetic alone, and no amount of regex tuning closes that gap; it needs an
issuer-prefix (IIN) check or context, which is a different kind of rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.contract import Document

# Priority order matters: a credit-card pattern ("digits and separators") is a superset of
# the SSN and phone shapes, so the specific patterns claim their spans first and the card
# rule only gets what's left. Detectors that skip this step double-count and report an
# inflated recall.
PII_TYPES = ("email", "ssn", "phone", "credit_card")

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),
    "phone": re.compile(
        r"(?<![\d-])(?:\+1[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\d-])"
    ),
    # 13–19 digits with optional single spaces/hyphens between them. Intentionally greedy:
    # the whole point is that the regex over-matches and Luhn does the discriminating.
    "credit_card": re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
}


def luhn_valid(number: str) -> bool:
    """Luhn mod-10 checksum. Non-digits are ignored; <12 digits is not a card."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class PIIFinding:
    pii_type: str
    start: int
    end: int
    text: str


class PIIDetector:
    """Regex detectors + optional Luhn gate on card candidates.

    `luhn=False` is not a straw man — it is what a PII redactor looks like before someone
    measures its false-positive rate. The bench runs both so the delta is a number.
    """

    name = "regex"

    def __init__(self, luhn: bool = True, types: tuple[str, ...] = PII_TYPES):
        unknown = set(types) - set(PII_TYPES)
        if unknown:
            raise ValueError(f"unknown pii types {sorted(unknown)}; expected {PII_TYPES}")
        self.luhn = luhn
        self.types = types

    def scan(self, text: str) -> list[PIIFinding]:
        findings: list[PIIFinding] = []
        claimed: list[tuple[int, int]] = []

        for pii_type in PII_TYPES:            # fixed priority, not caller order
            if pii_type not in self.types:
                continue
            for m in _PATTERNS[pii_type].finditer(text):
                start, end = m.start(), m.end()
                if any(start < c_end and end > c_start for c_start, c_end in claimed):
                    continue                  # a higher-priority type already owns this span
                if pii_type == "credit_card" and self.luhn and not luhn_valid(m.group(0)):
                    continue                  # digits that cannot be a card number
                claimed.append((start, end))
                findings.append(PIIFinding(pii_type, start, end, m.group(0)))

        return sorted(findings, key=lambda f: f.start)

    def has_pii(self, text: str) -> bool:
        return bool(self.scan(text))

    def redact_text(self, text: str) -> tuple[str, list[PIIFinding]]:
        """Replace every finding with `[REDACTED:TYPE]`. Returns (clean_text, findings)."""
        findings = self.scan(text)
        out, cursor = [], 0
        for f in findings:
            out.append(text[cursor : f.start])
            out.append(f"[REDACTED:{f.pii_type.upper()}]")
            cursor = f.end
        out.append(text[cursor:])
        return "".join(out), findings

    def redact_documents(self, docs: list[Document]) -> tuple[list[Document], dict[str, int]]:
        """Ingest-time redaction: PII never reaches the chunker, so it never reaches the index.

        Returns new Documents (never mutates the input) + a per-type count for the report.
        """
        counts: dict[str, int] = {t: 0 for t in self.types}
        clean_docs: list[Document] = []
        for d in docs:
            text, findings = self.redact_text(d.text)
            for f in findings:
                counts[f.pii_type] = counts.get(f.pii_type, 0) + 1
            metadata = dict(d.metadata)
            if findings:
                metadata["pii_redacted"] = len(findings)
            clean_docs.append(Document(doc_id=d.doc_id, text=text, metadata=metadata))
        return clean_docs, counts
