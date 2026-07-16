"""Security & governance (Phase 14) — the layer that assumes the corpus is hostile.

Every earlier phase optimizes for "did we retrieve the right chunk". This one asks the
questions a production deployment gets asked instead: what if a retrieved chunk contains
*instructions*? what if it contains a customer's card number? what if this user isn't
allowed to see it? what if the law says it must cease to exist?

Four defenses, each benchmarked on a labeled set rather than asserted:
- `injection` — detect/strip instructions hiding in RETRIEVED text (the indirect,
  second-order attack) + a spotlighting assembler that delimits untrusted data.
- `pii` — regex + **Luhn-validated** card detection; redact at ingest and at output.
- `acl` — permission-aware retrieval: enforce doc ACLs *at retrieval*, and prove that the
  post-hoc "filter after generation" alternative leaks.
- `rtbf` — right-to-be-forgotten: hard-delete a doc from the index + every cache, and prove
  a naive tombstone leaves it reachable.
- `audit` — append-only decision log (metadata only; prompts and chunk text never logged).

Registers `retriever: acl` and `assembler: screened|spotlight`. Not imported by
`modules/__init__.py` — import `modules.security` explicitly (these stages change what the
model is allowed to see, so they are opt-in, never ambient).
"""

from modules.security import acl  # noqa: F401
from modules.security import attacks  # noqa: F401
from modules.security import audit  # noqa: F401
from modules.security import injection  # noqa: F401
from modules.security import pii  # noqa: F401
from modules.security import rtbf  # noqa: F401
from modules.security.acl import ACLRetriever, Principal, can_read, postfiltered_answer
from modules.security.audit import AuditLog
from modules.security.injection import InjectionFinding, InjectionScreen
from modules.security.pii import PIIDetector, PIIFinding, luhn_valid
from modules.security.rtbf import AnswerCache, SecureStore

__all__ = [
    "ACLRetriever",
    "AnswerCache",
    "AuditLog",
    "InjectionFinding",
    "InjectionScreen",
    "PIIDetector",
    "PIIFinding",
    "Principal",
    "SecureStore",
    "can_read",
    "luhn_valid",
    "postfiltered_answer",
]
