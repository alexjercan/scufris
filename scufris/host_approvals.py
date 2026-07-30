"""The decision seam: ONE place where a host action is approved, denied or undone.

Two surfaces decide host actions - the dashboard (an operator session) and
Telegram (an allowlisted chat) - and "the same enforcement on both" is not a
promise anyone can keep by writing the rules twice. So the rules live here, once,
and each surface's only job is to say WHO is deciding
(``tasks/20260729-125040/DECISION.md`` section 3). The web routes and the bot
call the same methods; everything after the actor is derived - already decided,
expired, one-way acknowledgement, the apply, the audit-facing operator string, the
notification back to the requesting agent - is this module.

What still is NOT here, deliberately: the argv, the preview, the proposal state
machine and the audit log. Those are the root helper's
(``scufris.hostd``), and this module cannot reach the system except by asking it.
``apply`` is called from exactly one place - ``approve`` below - so an action with
no approval has no route to execution, not because a check refuses it but because
nothing else calls it.

The confirmation requirement itself lives with the RECORD it describes
(``host_actions.confirmation_for``), so the queue listing can carry it inline on
every row - see that function for the rule, and for the R1 carve-out that
MEASUREMENT forced into it.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Callable

from .host_actions import (
    AlreadyDecided,
    Confirmation,
    Decision,
    HostActionRecord,
    HostActionStore,
    UnknownAction,
    confirmation_for,
)
from .hostclient import (
    HostApplyError,
    HostApplyEvent,
    HostApplyResult,
    HostdClient,
    HostSupervisor,
)
from .hostd.audit import Requester
from .hostd.protocol import ProposalState, ProposalView, ResultFrame
from .supervisor import RunState

logger = logging.getLogger(__name__)


class ConfirmationRequired(ValueError):
    """A one-way action was approved through the ordinary confirmation.

    Raised BEFORE anything reaches the helper. The message names the token,
    because the caller that hits this is a surface that has not been taught the
    strong path yet - and the operator behind it should not be left guessing.
    """

    def __init__(self, confirmation: Confirmation) -> None:
        self.confirmation = confirmation
        super().__init__(
            "this action cannot be undone "
            f"({confirmation.undo}), so approving it needs the explicit "
            f"acknowledgement {confirmation.acknowledge!r} - the ordinary "
            "confirmation is refused here"
        )


class ProposalExpired(RuntimeError):
    """The proposal's window has closed, so the preview no longer describes a
    decision that can be made. Refused locally rather than sent to the helper: the
    helper would refuse it too, and the operator deserves the reason rather than a
    generic failure."""


class NotApplied(RuntimeError):
    """An undo was asked for on something that never ran."""


class CannotUndo(ValueError):
    """An undo was asked for on an action that declares it has none."""


class NoLiveRun(RuntimeError):
    """A cancel was asked for on an action with nothing in flight."""


# `HostApprovalService` defines a public `list()` method, which shadows the builtin
# `list` inside class-scope annotations (mypy resolves `list[X]` there to the
# method). This module-level alias, bound where `list` is still the builtin, lets
# those methods annotate a real list return - the same trick `agent_store` uses.
RecordList = list[HostActionRecord]


# A hook fires after a state change, for surfaces and for the requesting agent.
# Hooks are notified, never consulted: one raising must not fail a decision that
# has already happened, so every call is guarded.
Hook = Callable[[HostActionRecord], None]


def decision_message(record: HostActionRecord) -> str | None:
    """What to tell the agent that asked for this action, or None if there is
    nothing to tell it YET.

    A denial and a finished apply are both news the requesting agent needs: a
    denied agent that never hears the reason retries blindly, and an agent that
    never hears the result reports "I proposed it" as though that were the outcome.
    A bare "approved, starting" is NOT news - the result is moments away and two
    messages would cost the agent a turn to learn nothing - so this returns None
    for it and speaks once the apply is terminal.
    """
    proposal = record.proposal
    head = f"host action {record.id} ({proposal.summary})"
    if record.decision is Decision.DENIED:
        reason = record.reason.strip() or "(no reason given)"
        return (
            f"{head} was DENIED by {record.decided_by or 'the operator'}.\n"
            f"Reason: {reason}\n"
            "Do not propose the same action again unless that reason is addressed. "
            "Tell the operator what you would need instead, or propose something "
            "that answers the objection."
        )
    if record.decision is not Decision.APPROVED:
        return None
    if record.error:
        return (
            f"{head} was approved by {record.decided_by or 'the operator'}, but the "
            f"apply did not complete: {record.error}\n"
            "Re-read the host before assuming nothing happened."
        )
    result = record.result
    if result is None:
        return None  # approved and running; the result is the news, not this
    progress = (
        f", {result.steps_completed}/{result.steps_total} steps"
        if result.steps_total > 1
        else ""
    )
    outcome = "APPLIED" if result.ok else "FAILED"
    lines = [
        f"{head} was approved by {record.decided_by or 'the operator'} and "
        f"{outcome} ({result.outcome}, exit {result.returncode}{progress}).",
    ]
    if result.detail:
        lines.append(result.detail)
    if not result.ok and result.steps_total > 1:
        # A half-applied multi-step action is a state of its own, and the plan says
        # so in its own words - repeat them rather than paraphrasing.
        lines.append(
            "This action has several steps and did not finish all of them; read "
            f"host_action_status({record.id}) before acting on this."
        )
    if result.ok and result.reversal.possible:
        lines.append(f"It can be undone: {result.reversal.summary}")
    return "\n".join(lines)


class HostApprovalService:
    """Approve, deny, undo and cancel host actions - the ONE decision path.

    Holds the app-side registry, the helper client and the apply supervisor. Both
    surfaces construct nothing: they call these methods with an actor string they
    derived from their own credential.
    """

    def __init__(
        self,
        *,
        hostd: HostdClient,
        actions: HostActionStore,
        supervisor: HostSupervisor,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._hostd = hostd
        self._actions = actions
        self._supervisor = supervisor
        self._now = clock
        self._proposed_hooks: list[Hook] = []
        self._restored_hooks: list[Hook] = []
        self._decided_hooks: list[Hook] = []
        self._last_refresh = 0.0

    # --- wiring ----------------------------------------------------------

    def on_proposed(self, hook: Hook) -> None:
        """Notify ``hook`` when an action enters the queue (the requesting agent is
        marked as waiting on the operator; a surface announces it)."""
        self._proposed_hooks.append(hook)

    def on_restored(self, hook: Hook) -> None:
        """Notify ``hook`` when a proposal is recovered from the helper rather than
        newly made (see ``refresh_pending``).

        Separate from ``on_proposed`` because the two mean different things to
        different listeners: the requesting agent's BLOCKED marking wants BOTH (it
        is still waiting either way), while a surface that ANNOUNCES a new proposal
        wants only the first - re-announcing month-old news on every restart is how
        a notification channel gets muted.
        """
        self._restored_hooks.append(hook)

    def on_decided(self, hook: Hook) -> None:
        """Notify ``hook`` when an action is approved, denied, or has finished
        applying (the requesting agent is resumed with the outcome; a surface
        updates what it showed)."""
        self._decided_hooks.append(hook)

    def _fire(self, hooks: list[Hook], record: HostActionRecord) -> None:
        for hook in hooks:
            try:
                hook(record)
            except Exception:  # noqa: BLE001 - a listener must not fail a decision
                logger.exception("host approval hook failed for %s", record.id)

    # --- reads -----------------------------------------------------------

    @property
    def store(self) -> HostActionStore:
        return self._actions

    def get(self, action_id: str) -> HostActionRecord:
        """The record, or ``UnknownAction``."""
        return self._actions.get(action_id)

    def list(self) -> list[HostActionRecord]:
        return self._actions.list()

    def confirmation(self, action_id: str) -> Confirmation:
        return confirmation_for(self.get(action_id).proposal)

    def decidable(self) -> RecordList:
        """The actions a decision can still be MADE on, newest first.

        Narrower than "decision is pending": it also excludes a proposal the helper
        has moved on, one whose machine has drifted, and one whose window has closed.
        One definition, used by every surface that offers a control - a queue that
        offers a button the service will refuse is a queue that lies about what the
        operator can do.
        """
        live: RecordList = []
        for record in self._actions.list():
            try:
                self._refuse_undecidable(record)
            except (AlreadyDecided, ProposalExpired):
                continue
            live.append(record)
        return live

    def live_for_agent(self, agent_id: str) -> HostActionRecord | None:
        """The LIVE approval this agent is waiting on, or None.

        Live means a decision can still be made: undecided, still PENDING with the
        helper, and inside its window. That is a narrower question than "is this
        agent's outcome BLOCKED", and the difference is a real trap: nothing clears
        a BLOCKED outcome when a proposal simply EXPIRES (a decision clears it by
        resuming the agent, but an expiry resumes nobody), so a refusal keyed on the
        outcome state alone locks the agent out permanently after one proposal the
        operator never answered - unreachable by the orchestrator AND
        unacknowledgeable, with the approval that would have freed it no longer
        approvable. Review round 1, R1.1.

        Newest first, so an agent with two proposals reports the one it is actually
        waiting on.
        """
        if not agent_id:
            return None
        for record in self.decidable():
            if record.proposal.requester.agent == agent_id:
                return record
        return None

    # --- the decision path ------------------------------------------------

    def record_proposal(self, proposal: ProposalView) -> HostActionRecord:
        """Put a freshly created proposal in the queue and announce it.

        Called by the propose surfaces (the HTTP route and the config-change flow),
        so "a proposal exists" has one place that tells the requesting agent it is
        waiting on a human and tells the operator's surfaces to show it.
        """
        record = self._actions.put(proposal)
        self._fire(self._proposed_hooks, record)
        return record

    async def refresh_pending(self, *, min_interval: float = 0.0) -> int:
        """Rebuild the queue from the helper. Returns how many were recovered.

        The helper holds every proposal, so this - not a state file - is how the app
        gets a truthful queue back after a restart, and how it learns about a
        proposal made by another client of the same socket (the example script, a
        second app process). Only ADDITIONS are applied: the helper is asked what is
        still pending, and anything the app is missing is put in the queue with its
        original requester, which re-marks the agent that asked as waiting.

        The asymmetry is deliberate. A record the app holds that the helper no
        longer lists is NOT deleted here: the app cannot tell "expired" from "denied
        by the other surface" from "applied a moment ago" out of an absence, and the
        record carries its own expiry for the surfaces to render. The decision path
        refuses it either way (``_refuse_undecidable``), which is the check that
        matters.

        ``min_interval`` throttles: the queue listing calls this on every poll, and
        one socket round trip per poll per open tab is waste, not truth.
        """
        if not self._hostd.configured:
            return 0
        moment = self._now()
        if min_interval and moment - self._last_refresh < min_interval:
            return 0
        self._last_refresh = moment
        pending = await self._hostd.list_pending()
        recovered = 0
        for proposal in pending:
            try:
                self._actions.refresh(proposal.id, proposal)
            except UnknownAction:
                record = self._actions.put(proposal)
                recovered += 1
                self._fire(self._restored_hooks, record)
        if recovered:
            logger.info("recovered %d pending host action(s) from the helper", recovered)
        return recovered

    async def approve(
        self, action_id: str, *, actor: str, acknowledge: str = ""
    ) -> tuple[HostActionRecord, str]:
        """Approve an action and start it. Returns ``(record, run_id)``.

        The order is: refuse what cannot be approved, then claim the decision, then
        apply. Every refusal happens before the helper is touched, and the claim
        happens before the apply, so two surfaces racing on one id produce one
        execution and one ``AlreadyDecided`` rather than two applies.
        """
        record = self.get(action_id)
        self._refuse_undecidable(record)
        confirmation = confirmation_for(record.proposal)
        if confirmation.one_way and acknowledge != confirmation.acknowledge:
            raise ConfirmationRequired(confirmation)
        # Claims the decision; raises AlreadyDecided if the other surface won.
        self._actions.approve(action_id, operator=actor)
        run_id = f"host:{action_id}"

        async def _apply() -> AsyncIterator[HostApplyEvent]:
            async for event in self._hostd.apply(action_id, approved_by=actor):
                if isinstance(event, HostApplyResult):
                    self._finish(action_id, result=event.result)
                elif isinstance(event, HostApplyError):
                    self._finish(action_id, error=event.detail)
                yield event

        self._supervisor.start(
            run_id,
            _apply,
            # Serialized on one key: approvals queue rather than overlapping.
            serialize_key="host-actions",
            on_complete=lambda state: self._record_run(action_id, state),
        )
        record = self._actions.attach_run(action_id, run_id)
        self._fire(self._decided_hooks, record)
        return record, run_id

    async def deny(
        self, action_id: str, *, actor: str, reason: str = ""
    ) -> HostActionRecord:
        """Refuse a proposal, burning it.

        The HELPER burns it first, and only then is the local record marked. The
        other order left a proposal the app showed as denied while the helper still
        held it PENDING and appliable for the rest of its TTL, with
        ``AlreadyDecided`` blocking the retry that would have fixed it (20260729-125029
        review round 1, R1.9). The helper's state is what decides whether the action
        can still run, so it moves first.
        """
        record = self.get(action_id)
        if record.decision is not Decision.PENDING:
            raise AlreadyDecided(f"this action was already {record.decision}")
        proposal = await self._hostd.deny(action_id, operator=actor, reason=reason)
        record = self._actions.deny(action_id, operator=actor, reason=reason)
        record.proposal = proposal
        self._fire(self._decided_hooks, record)
        return record

    async def revert(self, action_id: str, *, actor: str) -> HostActionRecord:
        """Propose the INVERSE of an applied action.

        An undo is itself a host action: it gets its own preview and its own
        approval. Nothing is reverted here - the reversal is proposed, and the
        operator approves it like anything else.
        """
        record = self.get(action_id)
        reversal = record.proposal.reversal
        if not reversal.possible or reversal.kind is None:
            raise CannotUndo(reversal.summary or "this action cannot be undone")
        if record.result is None or not record.result.ok:
            raise NotApplied(
                "this action has not been applied, so there is nothing to undo"
            )
        proposal = await self._hostd.propose(
            reversal.kind, dict(reversal.args), Requester(actor=actor)
        )
        return self.record_proposal(proposal)

    def cancel(self, action_id: str) -> HostActionRecord:
        """Stop an apply that is running.

        Cancelling reaches the helper, which signals the whole process group and
        records the cancellation - so the outcome is a recorded fact rather than an
        unknown state. Whatever the command had already done still stands, and the
        record says so.
        """
        record = self.get(action_id)
        if record.run_id is None or not self._supervisor.cancel(record.run_id):
            raise NoLiveRun("this action has no live run to cancel")
        return record

    # --- edges ------------------------------------------------------------

    def _refuse_undecidable(self, record: HostActionRecord) -> None:
        """Refuse a decision that cannot honestly be made, before anything moves.

        Three ways a queued proposal stops being decidable, all of which the
        operator should be told about in those words rather than through a generic
        helper failure: it was already decided (possibly by the other surface a
        second ago), its window closed, or the helper has already moved it to a
        terminal state - including DRIFTED, where the preview describes a machine
        that has since changed and the honest answer is to look at a fresh one.
        """
        if record.decision is not Decision.PENDING:
            raise AlreadyDecided(
                f"this action was already {record.decision} by "
                f"{record.decided_by or 'someone'}"
            )
        proposal = record.proposal
        if proposal.state is ProposalState.DRIFTED:
            raise ProposalExpired(
                "this machine has changed since the preview was taken, so the "
                "preview no longer describes what would happen - propose it again "
                "and read the new one"
            )
        if proposal.state is not ProposalState.PENDING:
            raise ProposalExpired(
                f"the helper has already moved this proposal to {proposal.state}"
            )
        if proposal.expires_at and self._now() > proposal.expires_at:
            raise ProposalExpired(
                "this proposal has expired, so the preview no longer describes a "
                "decision that can be made - propose it again to get a fresh one"
            )

    def _finish(
        self, action_id: str, *, result: ResultFrame | None = None, error: str = ""
    ) -> None:
        try:
            self._actions.finish(action_id, result=result, error=error)
        except UnknownAction:  # pragma: no cover - reaped mid-apply
            logger.info("host action %s vanished mid-apply", action_id)

    def _record_run(self, action_id: str, state: RunState) -> None:
        """Carry a run's terminal state onto the record, then announce the outcome.

        A cancelled or crashed run must not leave the action looking merely
        decided: the helper has recorded what happened, and so does this. The
        decided hooks fire here too, because THIS is when the requesting agent can
        be told how its action actually turned out.
        """
        try:
            record = self._actions.get(action_id)
        except UnknownAction:
            return
        if record.result is None and not record.error:
            if state.cancelled:
                record.error = (
                    "cancelled mid-apply; the helper signalled the process group "
                    "and recorded it. Re-read the host before assuming nothing "
                    "happened."
                )
            elif state.error:
                record.error = state.error
        self._fire(self._decided_hooks, record)
