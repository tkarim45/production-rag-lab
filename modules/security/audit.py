"""Append-only audit log (Phase 14).

An audit log exists to answer "who saw what, when, and why" after the fact. The design rule
that matters: **it logs decisions, not content**. Prompts, chunk text, and answers never
enter it — an audit log that mirrors the corpus is a second copy of the data you are trying
to govern (and a second thing right-to-be-forgotten has to reach).

Append-only in the honest sense: `record` only appends, and `events` hands back a copy. This
is an in-memory reference implementation — a real deployment writes JSONL to
write-once storage. The mechanism is the lesson; the sink is an implementation detail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One governance decision. Note what is absent: no text, no prompt, no answer."""

    ts: float
    event: str                    # "retrieve" | "acl_deny" | "delete" | "redact" | "injection"
    principal: str | None = None
    query_id: str | None = None
    doc_ids: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "event": self.event,
            "principal": self.principal,
            "query_id": self.query_id,
            "doc_ids": list(self.doc_ids),
            "detail": dict(self.detail),
        }


class AuditLog:
    """In-memory append-only event log."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        event: str,
        *,
        principal: str | None = None,
        query_id: str | None = None,
        doc_ids: tuple[str, ...] | list[str] = (),
        **detail: Any,
    ) -> AuditEvent:
        ev = AuditEvent(
            ts=time.time(),
            event=event,
            principal=principal,
            query_id=query_id,
            doc_ids=tuple(doc_ids),
            detail=detail,
        )
        self._events.append(ev)
        return ev

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)          # copy: callers cannot mutate history

    def of_type(self, event: str) -> list[AuditEvent]:
        return [e for e in self._events if e.event == event]

    def __len__(self) -> int:
        return len(self._events)
