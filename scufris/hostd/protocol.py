"""The wire contract between scufris and the root helper.

Newline-delimited JSON over a unix socket. One request per connection; the
response is one or more frames, ending in exactly one terminal frame (``result``
or ``error``). Records are single-line, so a reader never has to buffer for a
frame boundary.

Two properties are enforced by the shape of these models rather than by checks
scattered through the server:

- **A caller names a verb, never a command.** There is no argv on any request.
  ``ActionKind`` is a closed enum, so an unknown verb fails to parse at this
  boundary and never reaches the code that builds commands.
- **Every request carries the shared secret.** Not a handshake that a later
  frame can skip - each frame is authenticated on its own, so a connection
  cannot be escalated after the fact.

The secret raises the bar against another process of the same user (notably the
agent CLI subprocesses scufris itself spawns, which run arbitrary shell). It is
NOT a boundary against a compromised operator account - see
``tasks/20260729-125029/DECISION.md``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .actions import ActionKind, RiskClass, Step
from .audit import AuditRecord, Requester
from .preview import Fingerprint, Preview, Reversal

PROTOCOL_VERSION = 1


class Verb(StrEnum):
    """What a caller may ask for. There is no shell verb, at any privilege."""

    HELLO = "hello"
    # Read-only: the proposals the helper still holds PENDING. The app rebuilds its
    # queue from this after a restart, so a proposal made minutes before one is not
    # stranded unapprovable for the rest of its TTL - the HELPER stays the single
    # source of truth for what has been proposed, rather than the app persisting a
    # second copy next to the root-owned audit log
    # (tasks/20260729-125040/DECISION.md section 4). It builds no argv, names no
    # proposal id, and changes nothing.
    LIST_PENDING = "list_pending"
    PROPOSE = "propose"
    APPLY = "apply"
    DENY = "deny"
    CANCEL = "cancel"
    AUDIT_TAIL = "audit_tail"


class ProposalState(StrEnum):
    """A proposal is used once. Every state but PENDING is terminal."""

    PENDING = "pending"
    # Claimed by an apply that is in flight. Not terminal, but not pending
    # either: it is what makes the check-and-claim atomic, so two approvals
    # racing on one id produce one execution and one refusal.
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    # The preview described a system that has since moved. Terminal, so the
    # operator re-proposes and looks at a fresh preview rather than approving
    # the old description of a new world.
    DRIFTED = "drifted"


class ErrorCode(StrEnum):
    """Why a request was refused, in a form the caller can act on."""

    UNAUTHORIZED = "unauthorized"
    BAD_REQUEST = "bad_request"
    REFUSED = "refused"  # the helper will not do this at all
    NOT_FOUND = "not_found"
    EXPIRED = "expired"
    DRIFTED = "drifted"  # the system moved after the preview
    ALREADY_USED = "already_used"
    INTERNAL = "internal"


class Request(BaseModel):
    """One request frame. Flat on purpose: it parses or it does not."""

    verb: Verb
    secret: str = ""
    # PROPOSE
    kind: ActionKind | None = None
    args: dict[str, object] = Field(default_factory=dict)
    requester: Requester = Field(default_factory=Requester)
    # APPLY / CANCEL. ``approved_by`` is the operator identity the APP reports
    # after its own session check; the helper records it and does not claim to
    # have verified it. What the helper verifies is that the action being
    # applied is the one it previewed.
    proposal_id: str = ""
    approved_by: str = ""
    reason: str = ""
    # AUDIT_TAIL
    limit: int = Field(default=50, ge=1, le=500)


class ProposalView(BaseModel):
    """A proposal as the app and the operator see it."""

    id: str
    kind: ActionKind
    risk: RiskClass
    args: dict[str, object] = Field(default_factory=dict)
    # Every command this action would run, in order. A caller approves THESE, not
    # a description of them.
    steps: list[Step] = Field(default_factory=list)
    summary: str
    preview: Preview
    reversal: Reversal
    fingerprint: Fingerprint
    requester: Requester = Field(default_factory=Requester)
    created_at: float
    expires_at: float
    state: ProposalState = ProposalState.PENDING


class HelloFrame(BaseModel):
    type: Literal["hello"] = "hello"
    version: int = PROTOCOL_VERSION
    verbs: list[ActionKind] = Field(default_factory=lambda: list(ActionKind))


class ProposalFrame(BaseModel):
    type: Literal["proposal"] = "proposal"
    proposal: ProposalView


class PendingFrame(BaseModel):
    """Every proposal the helper is still holding PENDING, oldest first."""

    type: Literal["pending"] = "pending"
    proposals: list[ProposalView] = Field(default_factory=list)


class OutputFrame(BaseModel):
    type: Literal["output"] = "output"
    stream: str
    text: str


class ResultFrame(BaseModel):
    """The terminal frame of an apply."""

    type: Literal["result"] = "result"
    proposal_id: str
    ok: bool
    outcome: str
    returncode: int | None = None
    duration_seconds: float = 0.0
    # How far a multi-step action got. `steps_completed < steps_total` on a
    # failure is a HALF-applied action, which for a configuration change means
    # this boot and the next boot disagree - see `Plan.partial_detail`.
    steps_completed: int = 0
    steps_total: int = 0
    reversal: Reversal
    detail: str = ""


class AuditFrame(BaseModel):
    type: Literal["audit"] = "audit"
    records: list[AuditRecord] = Field(default_factory=list)


class ErrorFrame(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    detail: str


Frame = (
    HelloFrame
    | ProposalFrame
    | PendingFrame
    | OutputFrame
    | ResultFrame
    | AuditFrame
    | ErrorFrame
)

TERMINAL_TYPES = frozenset(
    {"result", "error", "proposal", "pending", "audit", "hello"}
)


def encode(frame: BaseModel) -> bytes:
    """One frame as a single line of JSON."""
    return frame.model_dump_json().encode("utf-8") + b"\n"
