"""The host surface: metrics, the privileged action queue, and the checks.

propose -> preview -> approve -> apply -> audit -> roll back. The verbs, the
previews, the proposals and the audit log all live in the root helper
(`scufris.hostd`); the decision rules live in `HostApprovalService`; the clock
and the run bookkeeping live in `HostScheduler`. What is here is the
operator-facing HTTP surface over those - which is to say translation, and
nothing else.

The approval endpoints are in `auth.OPERATOR_ONLY_PATTERN`, so the machine bearer
token - which the app's own agent subprocesses hold - is refused there before the
middleware's short-circuit. An agent may propose. Only a human with a session may
approve.

`/api/config` rides along here rather than having a router of its own: it is one
route, and the settings router it belongs to has not landed yet (see this task's
DECISION.md).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..digest import Digest, DigestStore
from ..host import HostOverview
from ..host.overview import MIN_HOST_OVERVIEW_TTL, HostOverviewCache
from ..host_actions import (
    AlreadyDecided,
    Confirmation,
    HostActionRecord,
    HostActionStore,
    UnknownAction,
)
from ..host_approvals import (
    CannotPropose,
    CannotUndo,
    ConfirmationRequired,
    HostApprovalService,
    NoLiveRun,
    NotApplied,
    ProposalExpired,
)
from ..hostclient import HostdClient, HostdError, HostdUnavailable, HostSupervisor
from ..hostd.actions import ActionKind
from ..hostd.audit import AuditRecord
from ..metrics import Collector, HostStats
from ..processes import ProcessCollector, ProcessList
from ..scheduler import WATCH, HostScheduler, ScheduleState
from .auth import SessionGate
from .errors import hostd_http_error
from .sse import last_event_id, relay_bus_sse

logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    poll_seconds: float
    agent_enabled: bool
    # The dashboard polls the host overview on its own, much slower clock: that
    # endpoint shells out to systemctl and nixos-rebuild, and folding it into the
    # 2s stats poll would make the live gauges hostage to a subprocess.
    host_overview_seconds: float = 30.0


class HostActionRequest(BaseModel):
    """Propose a privileged host action.

    A verb and its typed arguments - never a command. The helper builds the
    argv, and an unknown verb fails to parse here rather than reaching it.
    """

    kind: ActionKind
    args: dict[str, object] = Field(default_factory=dict)
    # Which agent asked, when one did. Recorded in the audit; it does not grant
    # anything.
    agent: str = ""
    run: str = ""


class HostDecisionRequest(BaseModel):
    """The operator's answer to a proposal."""

    reason: str = ""


class HostApproveRequest(BaseModel):
    """The operator's approval, plus the acknowledgement a ONE-WAY action needs.

    Optional, because an ordinary (reversible) approval needs nothing beyond being
    the operator - but a proposal whose ``reversal.possible`` is false is refused
    (422) unless ``acknowledge`` carries the token
    ``host_approvals.confirmation_for`` names. That refusal lives in the service, so
    the web and Telegram surfaces cannot each decide what "are you sure" means.
    """

    acknowledge: str = ""


class DigestView(BaseModel):
    """What the dashboard shows about the scheduled checks.

    Module level like every other response model here, and not by preference: a
    response model defined INSIDE the router factory is a local whose forward
    reference FastAPI cannot resolve, and `app.openapi()` fails with "not fully
    defined".
    """

    schedules: list[ScheduleState]
    digests: list[Digest]
    muted_until: float
    # False when host checks are switched off entirely, so the page can say that
    # rather than rendering an empty history as if the scheduler were healthy.
    enabled: bool


class HostActionLaunched(BaseModel):
    """What an approval returns: the record plus the run carrying it out."""

    action: HostActionRecord
    run_id: str


@dataclass(frozen=True)
class HostDeps:
    """Everything the host routes read, named once and passed in.

    Frozen and explicit rather than closed over `create_app`'s scope: that scope
    is what made the routes untestable, and it is what let a route read a name
    bound hundreds of lines below it. Here a missing dependency is a construction
    error, at the include, rather than a NameError on the first request.
    """

    settings: Settings
    gate: SessionGate
    collector: Collector
    processes: ProcessCollector
    overview: HostOverviewCache
    hostd: HostdClient
    actions: HostActionStore
    approvals: HostApprovalService
    runs: HostSupervisor
    scheduler: HostScheduler
    digests: DigestStore


def build_host_router(deps: HostDeps) -> APIRouter:
    """The host metrics, action queue, checks and app-config routes."""
    router = APIRouter()

    async def _action_or_404(action_id: str) -> HostActionRecord:
        # Offloaded: the store opens a transaction, and every caller is a route.
        try:
            return await asyncio.to_thread(deps.actions.get, action_id)
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None

    async def _digest_view() -> DigestView:
        return DigestView(
            schedules=await deps.scheduler.states(),
            digests=await asyncio.to_thread(deps.digests.list),
            muted_until=deps.settings.host_digest_muted_until,
            enabled=deps.settings.host_checks_enabled,
        )

    @router.get("/api/stats")
    def get_stats() -> HostStats:
        """Return a fresh read-only snapshot of host metrics."""
        return deps.collector.sample()

    @router.get("/api/processes")
    def get_processes() -> ProcessList:
        """Return current processes aggregated by application."""
        return deps.processes.sample()

    @router.get("/api/host/overview")
    async def get_host_overview() -> HostOverview:
        """The host's inspection snapshot: failed units, generations, storage,
        thermals.

        Cached for ``host_overview_seconds`` and collected off the event loop:
        it runs subprocesses (systemctl, nixos-rebuild), which must never block
        the loop serving the live stats poll.
        """
        return await deps.overview.overview()

    @router.post("/api/host/actions", status_code=201)
    async def propose_host_action(
        body: HostActionRequest, request: Request
    ) -> HostActionRecord:
        """Ask the helper to preview an action. Proposing changes nothing.

        Open to an agent as well as the operator: proposing is how an assistant
        asks. It is the APPROVAL that is a human act, and that is a different
        endpoint with a different credential requirement.

        ``activate`` is refused, by the service rather than here - it is a rule
        about what may be proposed at all, and the MCP tool surface must not be
        able to reach a different answer.
        """
        try:
            return await deps.approvals.propose(
                body.kind,
                body.args,
                await deps.gate.requester_identity(
                    request, agent=body.agent, run=body.run
                ),
            )
        except CannotPropose as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise hostd_http_error(exc) from exc

    @router.get("/api/host/actions")
    async def list_host_actions() -> list[HostActionRecord]:
        """The proposal queue, newest first.

        Reconciles with the helper first (throttled), so the queue shows proposals
        this process did not create - one made before a restart, or by another client
        of the same socket - rather than only what it happens to remember.
        """
        try:
            await deps.approvals.refresh_pending(
                min_interval=deps.settings.host_queue_refresh_seconds
            )
        except (HostdUnavailable, HostdError) as exc:
            logger.debug("queue reconcile skipped: %s", exc)
        return await asyncio.to_thread(deps.actions.list)

    @router.get("/api/host/digests")
    async def get_host_digests() -> DigestView:
        """The recent digests and each schedule's last run.

        This is what makes silence readable: `watch` says nothing when there is
        nothing to say, so "did it fire" has to be answerable somewhere - here.
        """
        return await _digest_view()

    @router.post("/api/host/digests/run", status_code=202)
    async def run_host_checks_now(schedule: str = WATCH) -> DigestView:
        """Start one schedule's run now, and return immediately. OPERATOR ONLY.

        The "run it now" button, and the honest way to try a threshold change. It
        does NOT wait: a full pass walks the nix store and shells out to systemctl,
        which is tens of seconds of work that no HTTP request should hold a
        connection open for - the client polls `GET /api/host/digests` for the
        result, exactly as it polls an approved action's progress.
        """
        try:
            deps.scheduler.start_now(schedule)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await _digest_view()

    @router.get("/api/host/audit")
    async def get_host_audit(limit: int = 50) -> list[AuditRecord]:
        """The helper's own audit tail: what was requested, refused and applied."""
        try:
            return await deps.hostd.audit_tail(max(1, min(500, limit)))
        except (HostdUnavailable, HostdError) as exc:
            raise hostd_http_error(exc) from exc

    @router.get("/api/host/actions/{action_id}")
    async def get_host_action(action_id: str) -> HostActionRecord:
        return await _action_or_404(action_id)

    @router.get("/api/host/actions/{action_id}/events")
    async def host_action_events(
        action_id: str, http_request: Request
    ) -> StreamingResponse:
        """Relay an approved action's live output as SSE."""
        record = await _action_or_404(action_id)
        bus = deps.runs.bus(record.run_id) if record.run_id else None
        if bus is None:
            raise HTTPException(status_code=404, detail="this action has no live run")
        return relay_bus_sse(bus, last_event_id(http_request))

    @router.post("/api/host/actions/{action_id}/approve")
    async def approve_host_action(
        action_id: str, request: Request, body: HostApproveRequest | None = None
    ) -> HostActionLaunched:
        """Approve a previewed action and start it. OPERATOR ONLY.

        The decision itself is ``HostApprovalService.approve`` - shared with the
        Telegram surface - so what this route does is derive WHO is approving from
        the credential and translate the service's refusals into statuses: 409 for
        an action already decided (possibly by the other surface a moment ago) or a
        proposal whose window closed or whose machine drifted, 422 for a one-way
        action approved without its acknowledgement.
        """
        decision = body or HostApproveRequest()
        try:
            record, run_id = await deps.approvals.approve(
                action_id,
                actor=await deps.gate.operator_identity(request),
                acknowledge=decision.acknowledge,
            )
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None
        except ConfirmationRequired as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (AlreadyDecided, ProposalExpired) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise hostd_http_error(exc) from exc
        return HostActionLaunched(action=record, run_id=run_id)

    @router.get("/api/host/actions/{action_id}/confirmation")
    async def get_host_action_confirmation(action_id: str) -> Confirmation:
        """What approving this action requires: its risk class in words, the undo
        sentence (or the statement that there is none), and - for a one-way action -
        the acknowledgement token an approve must carry.

        Both approval surfaces render THIS rather than deciding for themselves how
        much friction an action deserves; it is also carried inline on every record
        in the queue listing, so a client needs this route only for a single action.
        """
        try:
            return await deps.approvals.confirmation(action_id)
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None

    @router.post("/api/host/actions/{action_id}/cancel")
    async def cancel_host_action(action_id: str) -> HostActionRecord:
        """Stop an apply that is running. OPERATOR ONLY.

        Cancelling reaches the helper, which signals the whole process group and
        records the cancellation - so the outcome is a recorded fact rather than
        an unknown state. Whatever the command had already done still stands,
        and the record says so.
        """
        try:
            return await deps.approvals.cancel(action_id)
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None
        except NoLiveRun as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/host/actions/{action_id}/deny")
    async def deny_host_action(
        action_id: str, body: HostDecisionRequest, request: Request
    ) -> HostActionRecord:
        """Refuse a proposal, burning it. OPERATOR ONLY.

        The reason is not decoration: it is what reaches the agent that asked, so
        it can adapt instead of proposing the same thing again.
        """
        try:
            return await deps.approvals.deny(
                action_id,
                actor=await deps.gate.operator_identity(request),
                reason=body.reason,
            )
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None
        except AlreadyDecided as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise hostd_http_error(exc) from exc

    @router.post("/api/host/actions/{action_id}/revert", status_code=201)
    async def revert_host_action(action_id: str, request: Request) -> HostActionRecord:
        """Propose the inverse of an applied action. OPERATOR ONLY.

        An undo is itself a host action: it gets its own preview and its own
        approval. Nothing is reverted by this call - the reversal is proposed,
        and the operator approves it like any other change.
        """
        try:
            return await deps.approvals.revert(
                action_id, actor=await deps.gate.operator_identity(request)
            )
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None
        except CannotUndo as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NotApplied as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise hostd_http_error(exc) from exc

    @router.get("/api/config")
    def get_config() -> AppConfig:
        """Client-facing knobs: poll intervals and whether the agent is on."""
        return AppConfig(
            poll_seconds=deps.settings.poll_seconds,
            agent_enabled=deps.settings.agent_enabled,
            # The floored value, so the client polls at the cadence the server
            # actually refreshes at rather than one the cache will not honour.
            host_overview_seconds=max(
                MIN_HOST_OVERVIEW_TTL, deps.settings.host_overview_seconds
            ),
        )

    return router


__all__ = [
    "AppConfig",
    "DigestView",
    "HostActionLaunched",
    "HostActionRequest",
    "HostApproveRequest",
    "HostDecisionRequest",
    "HostDeps",
    "build_host_router",
]
