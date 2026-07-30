"""The app's side of a host action: who asked, who decided, and what happened.

The privileged half of the record is the helper's (root-written, append-only,
independent of anything the app can rewrite). This is the OTHER half - the
request side the spike left as app state: what has been proposed, what the
operator decided, and which run is carrying it out, so the dashboard can show a
queue and a chat client can be told "waiting for you".

It is deliberately in-memory and bounded. A proposal is short-lived by
construction (the helper expires it in minutes), so durability here would buy
nothing the audit log does not already give - and the audit log is the record
that has to survive.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from enum import StrEnum
from typing import Callable

from pydantic import BaseModel

from .hostd.protocol import ProposalView, ResultFrame

# Bounded like any per-request registry: the oldest decided actions are dropped
# first, and a pending one is never dropped ahead of a decided one.
MAX_ACTIONS = 200


class Decision(StrEnum):
    """What the operator has said about a proposal, if anything yet."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class HostActionRecord(BaseModel):
    """One proposed action as the dashboard and the API see it."""

    proposal: ProposalView
    decision: Decision = Decision.PENDING
    decided_by: str = ""
    decided_at: float | None = None
    reason: str = ""
    # The supervisor run carrying out an approved action, so a client can attach
    # to its output stream.
    run_id: str | None = None
    result: ResultFrame | None = None
    error: str = ""

    @property
    def id(self) -> str:
        return self.proposal.id


class UnknownAction(KeyError):
    """No such action id in this app's registry."""


class AlreadyDecided(RuntimeError):
    """The operator has already answered this proposal.

    A second decision is refused rather than applied: an approval is a single
    act, and "approve twice" must not become "run twice" at any layer.
    """


class HostActionStore:
    """The app's bounded registry of proposed host actions."""

    def __init__(
        self,
        *,
        max_actions: int = MAX_ACTIONS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._actions: "OrderedDict[str, HostActionRecord]" = OrderedDict()
        self._max = max_actions
        self._now = clock

    def put(self, proposal: ProposalView) -> HostActionRecord:
        record = HostActionRecord(proposal=proposal)
        self._actions[proposal.id] = record
        self._reap()
        return record

    def get(self, action_id: str) -> HostActionRecord:
        try:
            return self._actions[action_id]
        except KeyError as exc:
            raise UnknownAction(action_id) from exc

    def list(self) -> list[HostActionRecord]:
        """Newest first - a queue is read from the top."""
        return list(reversed(self._actions.values()))

    def approve(self, action_id: str, *, operator: str) -> HostActionRecord:
        record = self._decide(action_id, Decision.APPROVED, operator=operator)
        return record

    def deny(
        self, action_id: str, *, operator: str, reason: str = ""
    ) -> HostActionRecord:
        record = self._decide(action_id, Decision.DENIED, operator=operator)
        record.reason = reason
        return record

    def _decide(
        self, action_id: str, decision: Decision, *, operator: str
    ) -> HostActionRecord:
        record = self.get(action_id)
        if record.decision is not Decision.PENDING:
            raise AlreadyDecided(
                f"this action was already {record.decision} by "
                f"{record.decided_by or 'someone'}"
            )
        record.decision = decision
        record.decided_by = operator
        record.decided_at = self._now()
        return record

    def attach_run(self, action_id: str, run_id: str) -> HostActionRecord:
        record = self.get(action_id)
        record.run_id = run_id
        return record

    def finish(
        self,
        action_id: str,
        *,
        result: ResultFrame | None = None,
        error: str = "",
    ) -> HostActionRecord:
        record = self.get(action_id)
        if result is not None:
            record.result = result
        if error:
            record.error = error
        return record

    def refresh(self, action_id: str, proposal: ProposalView) -> HostActionRecord:
        """Replace the held proposal snapshot (the helper owns its state)."""
        record = self.get(action_id)
        record.proposal = proposal
        return record

    def _reap(self) -> None:
        while len(self._actions) > self._max:
            for key, record in self._actions.items():
                if record.decision is not Decision.PENDING:
                    del self._actions[key]
                    break
            else:
                # Everything is pending: drop the oldest anyway rather than grow
                # without bound. The helper still holds it, so nothing is lost
                # that an audit read cannot recover.
                self._actions.popitem(last=False)


def render_action(record: HostActionRecord) -> str:
    """One action as plain text: what it does, what it would change, how to undo.

    Used by the propose endpoint's text rendering (so an agent shows the operator
    the real preview rather than its own paraphrase) and by
    ``examples/host_action.py``. The dashboard's own surface is
    20260729-125040; this is the text version that has to be right first.

    Two things are always present and never softened: the LABEL saying what kind
    of preview this is, and the reversal line saying how it can be undone or
    that it cannot.
    """
    proposal = record.proposal
    lines = [
        f"host action {proposal.id}",
        f"  what:     {proposal.summary}",
        f"  risk:     {proposal.risk} ({proposal.kind})",
    ]
    # Every command, in order. An action with two steps has two lines here, so
    # "what am I approving" is never a summary of a sequence.
    for index, step in enumerate(proposal.steps, start=1):
        prefix = f"  command {index}:" if len(proposal.steps) > 1 else "  command: "
        lines.append(f"{prefix} {' '.join(step.argv)}")
        if step.label and len(proposal.steps) > 1:
            lines.append(f"             ({step.label})")
    lines += [
        f"  decision: {record.decision}"
        + (f" by {record.decided_by}" if record.decided_by else ""),
        "",
        f"PREVIEW ({proposal.preview.kind}): {proposal.preview.label}",
    ]
    availability = proposal.preview.available.line()
    if availability:
        lines.append(f"  {availability}")
    lines.extend(f"  {line}" for line in proposal.preview.lines)
    lines.append("")
    if proposal.reversal.possible:
        lines.append(f"UNDO: {proposal.reversal.summary}")
    else:
        lines.append(f"NO UNDO: {proposal.reversal.summary}")
    if record.result is not None:
        outcome = "succeeded" if record.result.ok else "FAILED"
        progress = (
            f", {record.result.steps_completed}/{record.result.steps_total} steps"
            if record.result.steps_total > 1
            else ""
        )
        lines.append(
            f"RESULT: {outcome} ({record.result.outcome}, exit "
            f"{record.result.returncode}{progress})"
        )
        if record.result.detail:
            lines.append(f"  {record.result.detail}")
    if record.error:
        lines.append(f"ERROR: {record.error}")
    return "\n".join(lines)
