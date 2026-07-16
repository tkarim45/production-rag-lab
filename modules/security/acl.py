"""Permission-aware retrieval (Phase 14) — enforce ACLs where they are enforceable.

The claim this module exists to prove: **there is exactly one correct place to enforce a
document ACL, and it is before the model sees the chunk.**

The tempting alternative is post-hoc filtering — retrieve everything, generate, then strip
the citations the user isn't cleared for. It demos beautifully. The response contains no
forbidden citation, the compliance checkbox is green, and it is not access control. The
model *read the restricted chunk*, and the answer it wrote is derived from it. Stripping the
citation removes the evidence of the leak, not the leak. `postfiltered_answer` implements
this path faithfully so the bench can measure the secret coming out the other end.

ACL-at-retrieval has a cost, and pretending otherwise would be the same dishonesty in the
other direction: it needs **overfetch** (filter a pool, not a page — otherwise a user with
narrow permissions gets a short result list because their k slots were spent on documents
they can't see), and it defaults to **fail-closed**, which means an un-stamped document is
invisible. That default turns an ingestion gap into an availability bug instead of a
disclosure, which is the correct trade — but it is a real cost, and the bench reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.contract import Chunk, Query, Scored
from harness.registry import build, register
from modules.security.audit import AuditLog

PUBLIC = "public"


@dataclass(frozen=True)
class Principal:
    """Who is asking. `groups` is the set of ACL entries this caller satisfies."""

    user: str
    groups: frozenset[str] = field(default_factory=frozenset)
    tenant: str = "default"

    @property
    def grants(self) -> frozenset[str]:
        """Everything an ACL entry could name to admit this caller."""
        return frozenset({self.user, PUBLIC}) | self.groups

    @staticmethod
    def of(user: str, groups: list[str] | tuple[str, ...] = (), tenant: str = "default") -> "Principal":
        return Principal(user=user, groups=frozenset(groups), tenant=tenant)


def can_read(principal: Principal, chunk: Chunk, default: str = "deny") -> bool:
    """Doc-level authorization for one chunk.

    Policy, in order:
    1. **Tenant isolation first.** A tenant mismatch is a hard no regardless of the ACL —
       multi-tenancy is not an ACL entry you can be granted, it's a partition.
    2. `metadata["acl"]` names the principals/groups allowed (`"public"` admits everyone).
    3. No ACL stamped → `default`. Fail-closed ("deny") unless explicitly configured open.
    """
    md = chunk.metadata

    tenant = md.get("tenant")
    if tenant is not None and tenant != principal.tenant:
        return False

    acl = md.get("acl")
    if acl is None:
        return default == "allow"
    if isinstance(acl, str):
        acl = [acl]
    return bool(set(acl) & principal.grants)


@register("retriever", "acl")
class ACLRetriever:
    """Wrap any retriever; drop chunks the principal cannot read **before** ranking is cut.

    The principal is per-request state, so it is set on the instance (`set_principal`) the
    way a server constructs a request-scoped retriever — not smuggled through `Query`, which
    the contract deliberately keeps free of authorization concerns.
    """

    name = "acl"

    def __init__(
        self,
        base: str = "dense",
        principal: Principal | dict[str, Any] | None = None,
        default: str = "deny",
        overfetch: int = 4,
        audit: AuditLog | None = None,
        **base_params: Any,
    ):
        if default not in ("allow", "deny"):
            raise ValueError("default must be 'allow' (fail-open) or 'deny' (fail-closed)")
        self.default = default
        self.overfetch = max(1, int(overfetch))
        self.audit = audit
        self._base_name = base
        self._base_params = base_params
        self._base: Any = None
        self._principal = self._coerce(principal)

    @staticmethod
    def _coerce(p: Principal | dict[str, Any] | None) -> Principal | None:
        if p is None or isinstance(p, Principal):
            return p
        return Principal.of(p["user"], p.get("groups", ()), p.get("tenant", "default"))

    def set_principal(self, principal: Principal | dict[str, Any]) -> "ACLRetriever":
        self._principal = self._coerce(principal)
        return self

    def bind_corpus(self, chunks: list[Chunk], index: Any, embedder: Any) -> None:
        self._base = build("retriever", self._base_name, **self._base_params)
        if hasattr(self._base, "bind_corpus"):
            self._base.bind_corpus(chunks, index, embedder)
        elif hasattr(self._base, "bind"):
            self._base.bind(index)
        self._n_chunks = len(chunks)

    def retrieve(self, query: Query, k: int) -> list[Scored]:
        if self._base is None:
            raise RuntimeError("call bind_corpus(...) before retrieve()")
        if self._principal is None:
            # No principal is not "everyone" — it's a bug. Fail closed and say so.
            raise RuntimeError("ACLRetriever has no principal; call set_principal() per request")

        pool = self._base.retrieve(query, k * self.overfetch)
        allowed: list[Scored] = []
        denied: list[Scored] = []
        for s in pool:
            (allowed if can_read(self._principal, s.chunk, self.default) else denied).append(s)

        if denied and self.audit is not None:
            self.audit.record(
                "acl_deny",
                principal=self._principal.user,
                query_id=query.query_id,
                doc_ids=tuple(dict.fromkeys(s.chunk.doc_id for s in denied)),
                n_denied=len(denied),
            )
        return allowed[:k]


def postfiltered_answer(
    query: Query,
    scored: list[Scored],
    assembler: Any,
    generator: Any,
    principal: Principal,
    default: str = "deny",
) -> dict[str, Any]:
    """The **wrong** design, implemented honestly so it can be measured.

    Retrieve with no ACL → assemble everything → generate → *then* remove the citations the
    principal isn't cleared for. Returns the context the model actually saw and the answer it
    actually wrote, alongside the sanitized citation list, so a test can check whether the
    restricted text survived into the answer. It does.
    """
    context = assembler.assemble(query, scored)      # restricted chunks included
    gen = generator.generate(query, context)         # the model has now read them
    visible = [s for s in scored if can_read(principal, s.chunk, default)]
    return {
        "answer": gen["answer"],                     # NOT filtered — derived from everything
        "context": context,
        "visible_citations": [s.chunk.chunk_id for s in visible],
        "hidden_citations": [
            s.chunk.chunk_id for s in scored if not can_read(principal, s.chunk, default)
        ],
    }
