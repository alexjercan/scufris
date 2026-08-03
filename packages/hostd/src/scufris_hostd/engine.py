"""The helper's core: propose, apply, and the record of both.

The proposal registry lives HERE, in the privileged process, and that placement
is the point. A caller cannot hand the helper a command; it can only name a verb
and, later, an id the helper itself issued. So "preview one thing, apply
another" has no code path - the argv the operator approved is the argv that
runs, because there is only ever one of them and the helper is holding it.

Four refusals guard ``apply``, and each is a different failure:

- an id the helper never issued (**not_found**),
- a proposal that has already been used (**already_used**) - approvals do not
  replay,
- one whose window closed (**expired**),
- one whose preview described a system that has since moved (**drifted**).

The operator approval itself is enforced by the app: the socket is reachable
only by a caller holding the shared secret, and the only code path that calls
``apply`` is the session-authenticated approval endpoint. The helper records the
operator identity the app reports and does not pretend to have verified it -
what it verifies is that the action being applied is exactly the one it
previewed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable

from scufris_host import CommandResult, Outcome, Runner, run_command

from .actions import R3_KINDS, ActionKind, ActionRefused, Plan, build_plan
from .audit import AuditEvent, AuditLog, Requester
from .executor import Executor, OutputSink, run_action
from .files import Files, RealFiles
from .preview import build_preview, read_fingerprint
from .protocol import (
    AuditFrame,
    ErrorCode,
    HelloFrame,
    PendingFrame,
    ProposalState,
    ProposalView,
    ResultFrame,
)

logger = logging.getLogger(__name__)

# How long an operator has to look at a preview and decide. Long enough to read
# a closure diff, short enough that a proposal left open in a tab cannot be
# approved tomorrow against a system that has moved on.
DEFAULT_TTL_SECONDS = 600.0

# Proposals are bounded like any per-request registry: a helper that runs for
# months must not accumulate them. Terminal ones are dropped oldest-first past
# this cap.
MAX_PROPOSALS = 200

# Repeated identical refusals inside this window collapse into one record with a
# count. See HostdEngine.record_refusal: an unauthenticated caller can produce
# refusals as fast as it can open sockets, and an uncoalesced record per attempt
# lets it rotate the real history off disk.
REFUSAL_WINDOW_SECONDS = 60.0

# How many pending proposals one requester may hold at once. A proposal is cheap
# to hold but expensive to PRODUCE (the R2 previews walk the store under the
# global nix GC lock), so an unbounded propose path lets a caller with only the
# machine token keep that lock contended as root with nothing ever approved
# (review round 1, R1.10).
MAX_PENDING_PER_REQUESTER = 5


class _RefusalWindow:
    """How many identical refusals have arrived since the window opened."""

    __slots__ = ("started_at", "count")

    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self.count = 1


class HostdRefusal(Exception):
    """A request the helper declines, with the code the caller should see."""

    def __init__(self, code: ErrorCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class _Held:
    """A proposal plus the plan the helper keeps to itself."""

    __slots__ = ("view", "plan")

    def __init__(self, view: ProposalView, plan: Plan) -> None:
        self.view = view
        self.plan = plan


class HostdEngine:
    """Everything the helper does, with the socket peeled off.

    Driven directly by the tests and the example script, so the whole
    propose/preview/apply/audit path is exercisable without root, a socket, or a
    real host.
    """

    def __init__(
        self,
        audit: AuditLog,
        *,
        runner: Runner = run_command,
        executor: Executor = run_action,
        files: Files | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._audit = audit
        self._runner = runner
        self._executor = executor
        # The filesystem seam, for the questions R3 asks that no command answers
        # honestly (is this store path a NixOS system, what does this generation
        # link point at). Injectable for the same reason the runner is.
        self._files: Files = files if files is not None else RealFiles()
        self._ttl = ttl_seconds
        self._now = clock
        self._proposals: dict[str, _Held] = {}
        # Open refusal windows, keyed by the refusal's own text.
        self._refusals: dict[str, _RefusalWindow] = {}
        # Previews are serialized: they are the expensive half, and the R2 ones
        # take the global nix GC lock. One at a time means a burst queues instead
        # of multiplying the contention.
        self._preview_lock = asyncio.Lock()

    # --- reads -----------------------------------------------------------

    def hello(self) -> HelloFrame:
        return HelloFrame()

    def audit_tail(self, limit: int) -> AuditFrame:
        return AuditFrame(records=self._audit.tail(limit))

    def pending(self) -> PendingFrame:
        """The proposals still awaiting a decision, oldest first.

        Expires the stale ones FIRST, so this never reports a proposal as
        approvable when an apply would refuse it - the answer and the decision agree
        because they read the same state after the same sweep. Insertion order is
        creation order, so the oldest (the one closest to expiring) comes first.
        """
        self._expire_stale()
        return PendingFrame(
            proposals=[
                held.view
                for held in self._proposals.values()
                if held.view.state is ProposalState.PENDING
            ]
        )

    def proposal(self, proposal_id: str) -> ProposalView | None:
        held = self._proposals.get(proposal_id)
        return held.view if held is not None else None

    def record_refusal(self, detail: str) -> None:
        """Record something the helper refused before any action existed.

        COALESCED, and that is a security property rather than tidiness. An
        unauthenticated caller needs no secret to reach the socket, so a record
        per connection is a record an attacker can produce at will: measured,
        ~4000 connections a second, which rotates the entire audit history off
        disk in about ninety seconds at the production defaults. The record of
        what was done to this box would be destroyed by the one caller most
        likely to want that (review round 1, R1.2).

        So repeated refusals inside a window become ONE record carrying a count.
        The first refusal in a window is always written immediately - the signal
        an operator needs is "someone is trying", which the first one carries -
        and the flood that follows costs a bounded number of lines.
        """
        moment = self._now()
        window = self._refusals.get(detail)
        if window is not None and moment - window.started_at < REFUSAL_WINDOW_SECONDS:
            window.count += 1
            return
        # A closed window with more than the one already-recorded refusal leaves
        # a summary behind, so the count is never silently lost.
        if window is not None and window.count > 1:
            self._audit.record(
                AuditEvent.REFUSED,
                action_id="",
                detail=(
                    # count includes the one already written, so the summary
                    # reports what was SUPPRESSED rather than double-counting it.
                    f"{window.count - 1} further refusals in the "
                    f"{REFUSAL_WINDOW_SECONDS:.0f}s after: {detail}"
                ),
            )
        self._refusals = {
            key: held
            for key, held in self._refusals.items()
            if moment - held.started_at < REFUSAL_WINDOW_SECONDS
        }
        self._refusals[detail] = _RefusalWindow(started_at=moment)
        self._audit.record(AuditEvent.REFUSED, action_id="", detail=detail)

    # --- propose ---------------------------------------------------------

    async def propose(
        self,
        kind: ActionKind,
        args: dict[str, object],
        requester: Requester,
    ) -> ProposalView:
        """Validate an action, preview it, and hold it for approval.

        A proposal whose preview could not be produced is REFUSED rather than
        returned: the contract is propose -> preview -> approve, so an action
        the operator cannot see the consequences of must not become approvable.
        """
        self._expire_stale()
        self._refuse_a_flood_of_proposals(requester)
        try:
            plan = await asyncio.to_thread(
                build_plan, kind, args, runner=self._runner, files=self._files
            )
        except ActionRefused as exc:
            self._audit.record(
                AuditEvent.REFUSED,
                action_id="",
                kind=kind,
                args=dict(args),
                requester=requester,
                detail=str(exc),
            )
            raise HostdRefusal(ErrorCode.REFUSED, str(exc)) from exc

        async with self._preview_lock:
            preview, fingerprint, reversal = await asyncio.to_thread(
                build_preview, plan, self._runner, self._files
            )
        if not preview.ok:
            detail = (
                f"no preview could be produced: {preview.available.reason}. "
                "An action is not approvable without one."
            )
            self._audit.record(
                AuditEvent.REFUSED,
                action_id="",
                kind=kind,
                risk=plan.risk,
                args=plan.args,
                steps=plan.steps,
                requester=requester,
                detail=detail,
            )
            raise HostdRefusal(ErrorCode.REFUSED, detail)

        moment = self._now()
        view = ProposalView(
            id=uuid.uuid4().hex,
            kind=plan.kind,
            risk=plan.risk,
            args=plan.args,
            steps=plan.steps,
            summary=plan.summary,
            preview=preview,
            reversal=reversal,
            fingerprint=fingerprint,
            requester=requester,
            created_at=moment,
            expires_at=moment + self._ttl,
        )
        self._proposals[view.id] = _Held(view, plan)
        self._reap()
        self._audit.record(
            AuditEvent.REQUESTED,
            action_id=view.id,
            kind=view.kind,
            risk=view.risk,
            args=view.args,
            steps=view.steps,
            requester=requester,
            reversal=reversal.summary,
            restore_point=fingerprint.value,
            detail=plan.summary,
        )
        return view

    def deny(self, proposal_id: str, *, operator: str, reason: str = "") -> None:
        """Record an operator's refusal and burn the proposal."""
        held = self._require_pending(proposal_id)
        held.view.state = ProposalState.CANCELLED
        self._audit.record(
            AuditEvent.DENIED,
            action_id=proposal_id,
            kind=held.view.kind,
            risk=held.view.risk,
            args=held.view.args,
            steps=held.view.steps,
            requester=Requester(actor=operator),
            detail=reason,
        )

    # --- apply -----------------------------------------------------------

    async def apply(
        self,
        proposal_id: str,
        *,
        on_output: OutputSink,
        approved_by: str = "",
    ) -> ResultFrame:
        """Run a proposal the operator approved, once.

        The proposal is CLAIMED synchronously, before the first await, and only
        then is the drift check made. Two approvals racing on the same id
        therefore produce one execution and one ``already_used``: the loser is
        refused at the claim, not after the winner finishes.

        Claiming before the drift check rather than after is deliberate. The
        drift re-read is an await, and a check that spans an await is not a
        check - the first version of this method did exactly that and both
        racers ran the command.

        A plan's steps run IN ORDER and stop at the first failure. Where stopping
        halfway is itself a state the operator has to know about, the plan says so
        (``Plan.partial_detail``) and the record repeats it - "the second of two
        commands failed" must never be logged as "nothing happened".
        """
        held = self._require_pending(proposal_id)
        view = held.view
        view.state = ProposalState.APPLYING
        fingerprint = await asyncio.to_thread(
            read_fingerprint, held.plan, self._runner, self._files
        )
        if not fingerprint.matches(view.fingerprint):
            view.state = ProposalState.DRIFTED
            detail = (
                f"the system moved after the preview: {view.fingerprint.describes} "
                f"was {view.fingerprint.value or 'unreadable'} and is now "
                f"{fingerprint.value or 'unreadable'}. Propose it again and look "
                "at the new preview."
            )
            self._audit.record(
                AuditEvent.DENIED,
                action_id=proposal_id,
                kind=view.kind,
                risk=view.risk,
                args=view.args,
                steps=view.steps,
                requester=Requester(actor=approved_by),
                outcome="drifted",
                detail=detail,
            )
            raise HostdRefusal(ErrorCode.DRIFTED, detail)

        # The last thing checked before anything runs: a state in which starting
        # is destructive regardless of drift. R3 uses it for "a
        # switch-to-configuration is already in flight", where two interleaved
        # activations leave a system that matches neither configuration.
        blocker = await asyncio.to_thread(self._preflight, held.plan)
        if blocker:
            # Terminal, like drift, rather than back to PENDING. The app has
            # already recorded the operator's approval and refuses to record a
            # second one, so a proposal returned to PENDING here would be a
            # proposal only the helper thinks is still decidable. Propose it
            # again instead - for R3 that costs nothing, the build is done.
            view.state = ProposalState.CANCELLED
            self._audit.record(
                AuditEvent.DENIED,
                action_id=proposal_id,
                kind=view.kind,
                risk=view.risk,
                args=view.args,
                steps=view.steps,
                requester=Requester(actor=approved_by),
                outcome="blocked",
                detail=blocker,
            )
            raise HostdRefusal(ErrorCode.REFUSED, blocker)

        self._audit.record(
            AuditEvent.APPROVED,
            action_id=proposal_id,
            kind=view.kind,
            risk=view.risk,
            args=view.args,
            steps=view.steps,
            requester=Requester(actor=approved_by),
            reversal=view.reversal.summary,
            restore_point=view.fingerprint.value,
        )

        started = time.monotonic()
        completed = 0
        try:
            result, completed = await self._run_steps(held.plan, on_output)
        except asyncio.CancelledError:
            view.state = ProposalState.CANCELLED
            self._audit.record(
                AuditEvent.CANCELLED,
                action_id=proposal_id,
                kind=view.kind,
                risk=view.risk,
                args=view.args,
                steps=view.steps,
                requester=Requester(actor=approved_by),
                duration_seconds=time.monotonic() - started,
                outcome="cancelled",
                reversal=view.reversal.summary,
                restore_point=view.fingerprint.value,
                detail=(
                    "cancelled mid-apply: the process group was signalled. Any "
                    "work it had already done stands - re-read the host before "
                    "assuming the action did not happen."
                    + (f" {held.plan.cancel_detail}" if held.plan.cancel_detail else "")
                ),
            )
            raise

        ok = result.outcome is Outcome.OK
        view.state = ProposalState.APPLIED if ok else ProposalState.FAILED
        total = len(held.plan.steps)
        detail = "" if ok else result.reason()
        if not ok and completed > 0 and held.plan.partial_detail:
            # The failure is not "nothing happened": earlier steps succeeded.
            # Say which, and what that leaves behind, in the record itself.
            detail = (
                f"step {completed + 1} of {total} failed after {completed} "
                f"succeeded. {result.reason()} {held.plan.partial_detail}"
            )
        self._audit.record(
            AuditEvent.APPLIED if ok else AuditEvent.FAILED,
            action_id=proposal_id,
            kind=view.kind,
            risk=view.risk,
            args=view.args,
            steps=view.steps,
            requester=Requester(actor=approved_by),
            outcome=str(result.outcome),
            returncode=result.returncode,
            duration_seconds=result.duration_seconds,
            steps_completed=completed,
            reversal=view.reversal.summary,
            restore_point=view.fingerprint.value,
            detail=detail,
        )
        return ResultFrame(
            proposal_id=proposal_id,
            ok=ok,
            outcome=str(result.outcome),
            returncode=result.returncode,
            duration_seconds=result.duration_seconds,
            steps_completed=completed,
            steps_total=total,
            reversal=view.reversal,
            detail=detail,
        )

    async def _run_steps(
        self, plan: Plan, on_output: OutputSink
    ) -> tuple[CommandResult, int]:
        """Run a plan's steps in order, stopping at the first failure.

        Returns the result that decides the outcome - the failing step, or the
        last one when they all succeeded - and how many steps COMPLETED, which is
        what the record needs to describe a half-applied action.
        """
        completed = 0
        result = CommandResult(argv=[], outcome=Outcome.FAILED, stderr="no steps")
        for index, step in enumerate(plan.steps):
            if step.label:
                # The operator is watching a root command; say which one starts.
                on_output("stdout", f"[{index + 1}/{len(plan.steps)}] {step.label}\n")
            result = await self._executor(
                step.argv, timeout=step.timeout, on_output=on_output
            )
            if result.outcome is not Outcome.OK:
                return result, completed
            completed += 1
        return result, completed

    def _preflight(self, plan: Plan) -> str:
        """Why this plan must not start right now, or empty when it may.

        Distinct from drift on purpose. Drift asks "is the world still the one
        that was previewed"; this asks "is the world in a state where STARTING is
        itself harmful". Only R3 has such a state today.
        """
        if plan.kind in R3_KINDS:
            from . import nixos

            return nixos.switch_in_flight(self._runner)
        return ""

    # --- registry housekeeping -------------------------------------------

    def _refuse_a_flood_of_proposals(self, requester: Requester) -> None:
        """Cap the pending proposals one requester may hold.

        Proposing is meant to be free of consequence, and it is - but it is not
        free of COST: the R2 previews walk the whole store under the global nix
        GC lock. Without a cap, a caller holding only the machine token can loop
        `propose` and keep that lock contended as root while never approving
        anything (review round 1, R1.10).

        The key is `actor` ALONE, which the app derives from the credential. It
        is deliberately not `agent`: that is a body field the caller fills in, so
        keying on it let a machine caller vary the name per request and hold
        twenty pending proposals against a cap of five (review round 2, R2.2).
        The caller does not get to pick the identity a limit is counted against.
        """
        who = requester.actor
        pending = sum(
            1
            for held in self._proposals.values()
            if held.view.state is ProposalState.PENDING
            and held.view.requester.actor == who
        )
        if pending >= MAX_PENDING_PER_REQUESTER:
            raise HostdRefusal(
                ErrorCode.REFUSED,
                f"{who} already has {pending} proposals waiting for a decision "
                f"(the cap is {MAX_PENDING_PER_REQUESTER}). Previewing is not "
                "free - the cleanup previews walk the store under the nix GC "
                "lock - so decide on those before asking for more.",
            )

    def _require_pending(self, proposal_id: str) -> _Held:
        self._expire_stale()
        held = self._proposals.get(proposal_id)
        if held is None:
            # Deliberately the same answer for "never existed" and "reaped": an
            # id is not an oracle.
            raise HostdRefusal(
                ErrorCode.NOT_FOUND, "no such proposal, or it is no longer held"
            )
        if held.view.state is ProposalState.EXPIRED:
            raise HostdRefusal(
                ErrorCode.EXPIRED,
                "this proposal's approval window closed; propose it again and "
                "look at a fresh preview",
            )
        if held.view.state is not ProposalState.PENDING:
            raise HostdRefusal(
                ErrorCode.ALREADY_USED,
                f"this proposal is already {held.view.state}; an approval is "
                "used once and does not replay",
            )
        return held

    def _expire_stale(self) -> None:
        moment = self._now()
        for held in self._proposals.values():
            if (
                held.view.state is ProposalState.PENDING
                and moment > held.view.expires_at
            ):
                held.view.state = ProposalState.EXPIRED
                self._audit.record(
                    AuditEvent.EXPIRED,
                    action_id=held.view.id,
                    kind=held.view.kind,
                    risk=held.view.risk,
                    args=held.view.args,
                    steps=held.view.steps,
                    requester=held.view.requester,
                    detail="the approval window closed with no decision",
                )

    def _reap(self) -> None:
        """Drop the oldest terminal proposals past the cap.

        Insertion order is creation order, so the first terminal entries found
        are the oldest.
        """
        while len(self._proposals) > MAX_PROPOSALS:
            for key, held in self._proposals.items():
                if held.view.state is not ProposalState.PENDING:
                    del self._proposals[key]
                    break
            else:
                return
