"""Running one agent: launch, cancel, watch, message, signal, and read back.

The record surface is `api/agents.py`; this is everything under
`/api/agents/{id}/` that is about a TURN rather than about the row. The whole run
lifecycle - the one-run-per-agent guard, the supervised background job, the
sub-agent signals, the completion fan-out - lives in `AgentRunService`, and the
per-agent diagnostics reads live in `AgentDiagnostics`. What is here is
translation: which service is asked, and which status its refusal becomes.

The `require_agent*` / `launch` / `drain_turn` helpers are exported because
`api/agents.py` and `api/legacy_agent.py` translate the SAME refusals, and a
second copy is how two surfaces start answering different statuses for one
service error.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scufris_core import EventBus
from scufris_hostctl import HostApprovalService

from ..agent import StreamDone, StreamEvent
from ..agent_diagnostics import AccountInfo, AgentDiagnostics
from ..agent_store import ORCHESTRATOR_ID, AgentNotFound, AgentRecord, AgentStore
from ..backends import Capability, get_backend
from ..config import Settings
from ..enums import AgentState
from ..env_bridge import ensure_den_path
from ..health import AgentHealth
from ..mcp_models import AgentTool, McpServerHealth
from ..orchestrator import (
    AgentProjectMissing,
    AgentRunService,
    NoActiveRun,
    OrchestratorError,
)
from ..projects import Project
from ..sessions import MemoryFootprint, UsageQuota
from ..supervisor import AgentSupervisor
from .auth import SessionGate
from .errors import orchestrator_http_error
from .models import TranscriptResponse
from .sse import last_event_id, relay_bus_sse


class AgentRunRequest(BaseModel):
    # An optional goal override; falls back to the agent's stored goal.
    goal: str | None = None
    # The orchestrator chat that spawned this run (part 3), so a later
    # request_input routes back to it. Absent for a UI-launched run (unattributed).
    parent_session_id: str | None = None


class AgentChatRequest(BaseModel):
    # One user turn of a per-agent conversation.
    message: str
    # The orchestrator chat that sent this turn (part 3); see AgentRunRequest.
    parent_session_id: str | None = None


class AgentRequestInput(BaseModel):
    # A sub-agent signalling it is blocked and needs a decision (BC2). The
    # question the orchestrator must answer before the agent can continue.
    question: str


class RequestInputResult(BaseModel):
    agent_id: str
    state: AgentState


class AgentReportBack(BaseModel):
    # A sub-agent signalling it has FINISHED its task and is handing back a result.
    # The summary the orchestrator reads before acknowledging the agent.
    summary: str


class ReportBackResult(BaseModel):
    agent_id: str
    state: AgentState


class AcknowledgeResult(BaseModel):
    agent_id: str
    acknowledged: bool


class AgentForkRequest(BaseModel):
    # Revert-fork a single-session agent: rewind its one session to
    # ``message_index`` and continue from the edited ``text``.
    message_index: int
    text: str


class RunStarted(BaseModel):
    agent_id: str
    state: str


class CancelResult(BaseModel):
    agent_id: str
    # True if a live run was cancelled; False if the agent had no active run
    # (idempotent - a stop on an already-finished turn is not an error).
    cancelled: bool


class AgentRunStatus(BaseModel):
    """The merged live run-state (Supervisor) + rollout/session progress
    (AgentBackend.read_status) for one agent."""

    agent_id: str
    state: str
    session_id: str | None = None
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_window: int = 0
    last_message: str | None = None
    updated_at: float | None = None
    # The in-flight turn's prompt (steering stripped), set only while the run is
    # queued/running, so a client reattaching mid-turn renders the user bubble the
    # backend's durable log has not caught up on yet. None when idle/finished.
    prompt: str | None = None


def require_agent(runs: AgentRunService, agent_id: str) -> AgentRecord:
    """The agent, or the 404 every surface that names one answers with."""
    try:
        return runs.require_agent(agent_id)
    except AgentNotFound as exc:
        raise HTTPException(status_code=404, detail="no such agent") from exc


async def require_agent_async(runs: AgentRunService, agent_id: str) -> AgentRecord:
    """The same lookup, offloaded, for the routes that are `async def`.

    A store read takes SQLite's write lock, and taking it on the loop thread
    stalls every other request in the process
    (packages/core/src/scufris_core/engine.py).
    """
    return await asyncio.to_thread(require_agent, runs, agent_id)


def require_agent_project(runs: AgentRunService, agent: AgentRecord) -> Project | None:
    try:
        return runs.require_agent_project(agent)
    except AgentProjectMissing as exc:
        raise orchestrator_http_error(exc) from exc


async def require_agent_project_async(
    runs: AgentRunService, agent: AgentRecord
) -> Project | None:
    """The same lookup, offloaded, for the routes that are `async def`."""
    return await asyncio.to_thread(require_agent_project, runs, agent)


async def launch(
    runs: AgentRunService,
    agent: AgentRecord,
    project: Project | None,
    prompt: str,
    *,
    image_paths: list[str] | None = None,
    on_done: Callable[[], None] | None = None,
) -> tuple[str, EventBus[StreamEvent]]:
    """`AgentRunService.launch` with its refusals translated to statuses.

    The routes launch through this rather than through the service directly, so
    the 409/503/422 a client sees is decided in ONE place.
    """
    try:
        return await runs.launch(
            agent, project, prompt, image_paths=image_paths, on_done=on_done
        )
    except OrchestratorError as exc:
        raise orchestrator_http_error(exc) from exc


async def drain_turn(runs: AgentRunService, bus: EventBus[StreamEvent]) -> StreamDone:
    """`AgentRunService.drain` with its refusals translated to statuses."""
    try:
        return await runs.drain(bus)
    except OrchestratorError as exc:
        raise orchestrator_http_error(exc) from exc


@dataclass(frozen=True)
class AgentRunDeps:
    """What running an agent reads.

    ``approvals`` is here for one rule and it is not decoration: an agent waiting
    on a LIVE host approval is waiting for the OPERATOR, so the orchestrator may
    neither message it nor acknowledge the signal away.
    """

    settings: Settings
    agents: AgentStore
    runs: AgentRunService
    diagnostics: AgentDiagnostics
    gate: SessionGate
    approvals: HostApprovalService
    supervisor: AgentSupervisor


def build_agent_run_router(deps: AgentRunDeps) -> APIRouter:
    """Launch, cancel, watch, message and read back one agent's turns."""
    router = APIRouter()

    @router.post("/api/agents/{agent_id}/run")
    async def run_agent(agent_id: str, req: AgentRunRequest) -> RunStarted:
        """Launch a supervised background run for the agent, scoped to its project
        cwd via its configured backend. 404 unknown, 422 no goal / missing project,
        409 a run is already active.

        Async so it runs on the event loop thread - the supervisor schedules the
        background run via ``asyncio.create_task``, which needs a running loop (a
        sync endpoint runs in a worker thread with none)."""
        agent = await require_agent_async(deps.runs, agent_id)
        goal = (req.goal if req.goal is not None else agent.goal).strip()
        if not goal:
            raise HTTPException(
                status_code=422, detail="agent has no goal; provide one to run"
            )
        project = await require_agent_project_async(deps.runs, agent)
        if req.parent_session_id:
            # Stamp the child with the orchestrator chat that spawned it (part 3),
            # so a later request_input routes back to that chat.
            await asyncio.to_thread(
                deps.agents.record_spawn_parent,
                agent_id,
                ORCHESTRATOR_ID,
                req.parent_session_id,
            )
        run_id, _bus = await launch(deps.runs, agent, project, goal)
        # Report the supervisor's actual state (usually "queued" until a slot is
        # free), not an assumed "running".
        started = deps.supervisor.status(run_id)
        return RunStarted(
            agent_id=agent_id, state=started.state if started is not None else "running"
        )

    @router.post("/api/agents/{agent_id}/cancel")
    async def cancel_agent_run(agent_id: str) -> CancelResult:
        """Cancel the agent's in-flight run (the chat stop button, or the
        orchestrator's ``cancel_agent`` tool). Truly aborts the backend turn -
        the supervisor cancels the run task, whose drain aclose()s the backend
        stream so its cleanup runs (e.g. the Claude subprocess is killed). The
        persist callback then records a CANCELLED terminal outcome. Works for the
        orchestrator too (it is an agent in ``agent_runs`` keyed ORCHESTRATOR_ID).
        404 unknown agent, or 404 when the agent has no active run (mirroring
        ``/events``). Async: cancelling a task touches the running loop.
        """
        await require_agent_async(deps.runs, agent_id)
        try:
            deps.runs.cancel(agent_id)
        except NoActiveRun as exc:
            raise orchestrator_http_error(exc) from exc
        return CancelResult(agent_id=agent_id, cancelled=True)

    @router.get("/api/agents/{agent_id}/status")
    def agent_run_status(agent_id: str) -> AgentRunStatus:
        """Merge the live Supervisor run-state with the backend's read-only
        rollout/session progress for the agent."""
        status = deps.runs.status(require_agent(deps.runs, agent_id))
        result = AgentRunStatus(
            agent_id=status.agent_id,
            state=status.state,
            session_id=status.session_id,
            prompt=status.prompt,
        )
        progress = status.progress
        if progress is not None:
            result.turns = progress.turns
            result.tool_calls = progress.tool_calls
            result.input_tokens = progress.input_tokens
            result.output_tokens = progress.output_tokens
            result.context_window = progress.context_window
            result.last_message = progress.last_message
            result.updated_at = progress.updated_at
        return result

    @router.get("/api/agents/{agent_id}/events")
    async def agent_events(agent_id: str, http_request: Request) -> StreamingResponse:
        """Relay the agent's current run event bus as SSE (drop-safe; a reconnect
        replays via Last-Event-ID). 404 when the agent has no live run bus."""
        await require_agent_async(deps.runs, agent_id)
        try:
            bus = deps.runs.bus(agent_id)
        except NoActiveRun as exc:
            raise orchestrator_http_error(exc) from exc
        return relay_bus_sse(bus, last_event_id(http_request))

    @router.post("/api/agents/{agent_id}/chat")
    async def agent_chat(
        agent_id: str, req: AgentChatRequest, request: Request
    ) -> StreamingResponse:
        """Stream one chat turn with the agent as SSE, resuming its one session.
        Runs over the SAME supervisor + event bus + agent-run registry as a goal
        run, so ``/status`` and ``/events`` reflect the turn and a concurrent
        turn is refused (409). 404 unknown agent, 422 empty message / missing
        project. The persist callback writes the (possibly new) session id back,
        so the next turn resumes it.

        409 when an AGENT-credential caller (the orchestrator's ``message_agent``)
        messages an agent with a LIVE host approval outstanding. That agent is not
        waiting for the orchestrator, and a resume carrying "approved, go ahead"
        would be an answer the orchestrator has no authority to give - the operator
        decides, and the decision resumes the agent itself. The operator's own
        session may message it: reading its own chat is not deciding.

        LIVE, not merely BLOCKED: once the proposal is decided or its window has
        closed there is nothing for the orchestrator to interfere with, and refusing
        anyway would leave the agent unreachable for good (review round 1, R1.1).
        """
        agent = await require_agent_async(deps.runs, agent_id)
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        live = await deps.approvals.live_for_agent(agent_id)
        if live is not None and await deps.gate.caller_is_agent(request):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"agent {agent_id} is waiting for the OPERATOR to decide host "
                    f"action {live.id}, not for you. You cannot approve or deny it; "
                    "the operator does that in the dashboard or over Telegram, and "
                    "the decision resumes the agent with the outcome (or the denial "
                    "reason). Report that it is waiting instead of answering it."
                ),
            )
        project = await require_agent_project_async(deps.runs, agent)
        if req.parent_session_id:
            # Stamp the child with the orchestrator chat that sent this turn
            # (part 3), so a later request_input routes back to that chat.
            await asyncio.to_thread(
                deps.agents.record_spawn_parent,
                agent_id,
                ORCHESTRATOR_ID,
                req.parent_session_id,
            )
        _run_id, bus = await launch(deps.runs, agent, project, message)
        return relay_bus_sse(bus)

    @router.post("/api/agents/{agent_id}/request_input")
    def agent_request_input(
        agent_id: str, req: AgentRequestInput
    ) -> RequestInputResult:
        """A sub-agent signals it is blocked and needs a decision (BC2). Records a
        WAITING outcome carrying the question, keyed to the agent's CURRENT run so
        the turn-end completion preserves it (see ``AgentStore.request_input`` /
        ``mark_finished``); returns immediately - the agent ends its turn and the
        orchestrator answers later by resuming. 404 unknown agent (incl. the
        orchestrator, which is not a sub-agent), 422 empty question."""
        agent = require_agent(deps.runs, agent_id)
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        try:
            state = deps.runs.request_input(agent, question)
        except AgentNotFound as exc:
            # The orchestrator resolves via require_agent but is not a sub-agent
            # (no ``agents`` row), so request_input rejects it - surface as 404.
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return RequestInputResult(agent_id=agent_id, state=state)

    @router.post("/api/agents/{agent_id}/report_back")
    def agent_report_back(agent_id: str, req: AgentReportBack) -> ReportBackResult:
        """A sub-agent signals it has FINISHED its task and hands back a result.
        Records a REPORTED outcome carrying the summary, keyed to the agent's
        CURRENT run so the turn-end completion preserves it (see
        ``AgentStore.report_back`` / ``mark_finished``); returns immediately - the
        agent ends its turn and the orchestrator is woken / sees it in
        `/api/agents/pending`, reads the report and acknowledges (no resume). 404
        unknown agent (incl. the orchestrator, which is not a sub-agent), 422 empty
        summary."""
        agent = require_agent(deps.runs, agent_id)
        summary = req.summary.strip()
        if not summary:
            raise HTTPException(status_code=422, detail="summary must not be empty")
        try:
            state = deps.runs.report_back(agent, summary)
        except AgentNotFound as exc:
            # The orchestrator resolves via require_agent but is not a sub-agent
            # (no ``agents`` row), so report_back rejects it - surface as 404.
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return ReportBackResult(agent_id=agent_id, state=state)

    @router.post("/api/agents/{agent_id}/acknowledge")
    async def agent_acknowledge(agent_id: str) -> AcknowledgeResult:
        """Mark an agent's pending signal handled (BC3), so it drops out of
        `/api/agents/pending`. Idempotent: `acknowledged` is False if there was
        nothing pending (already handled, or no outcome). No 404 - a cleared or
        never-seen agent simply acks to False.

        A LIVE host approval is the one signal this cannot clear: it is the
        operator's to answer, and hiding it from the queue would hide a decision
        nobody has made. Once that approval is decided or expired the signal
        acknowledges like any other - a proposal the operator never answered must not
        leave the agent with an outcome that can never be cleared (review round 1,
        R1.1)."""
        if await deps.approvals.live_for_agent(agent_id) is not None:
            return AcknowledgeResult(agent_id=agent_id, acknowledged=False)
        # `async def` because the live-approval check reads the action store; both
        # store calls are therefore offloaded.
        return AcknowledgeResult(
            agent_id=agent_id, acknowledged=await deps.runs.acknowledge(agent_id)
        )

    @router.post("/api/agents/{agent_id}/fork")
    async def agent_fork(agent_id: str, req: AgentForkRequest) -> StreamingResponse:
        """Revert-fork a single-session agent and stream the continuation.

        A project agent keeps ONE session, so "forking" a past message rewinds
        that session to the fork point and continues from the edit: read the
        agent's transcript, seed a turn from ``messages[:message_index]`` + the
        edited text, and launch it against a session-cleared copy of the record so
        the seed opens a FRESH session. The persist callback writes that new
        session back as the agent's sole session, dropping the old tail (the
        revert). Streams SSE exactly like ``/chat``. 404 unknown, 422 empty text /
        missing project, 409 active or the orchestrator (which keeps its
        multi-session ``/api/agent/session/fork`` instead)."""
        agent = await require_agent_async(deps.runs, agent_id)
        if agent.id == ORCHESTRATOR_ID:
            raise HTTPException(
                status_code=409,
                detail="the orchestrator forks via /api/agent/session/fork",
            )
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="message must not be empty")
        project = await require_agent_project_async(deps.runs, agent)
        seed = await asyncio.to_thread(
            deps.runs.fork_seed, agent, agent.session_id, req.message_index, text
        )
        # Launch against a session-cleared copy so the seed opens a fresh session
        # (the revert). The turn still runs under the real agent id, so the persist
        # callback writes the new session id back to the actual record.
        reverted = agent.model_copy(update={"session_id": None})
        _run_id, bus = await launch(deps.runs, reverted, project, seed)
        return relay_bus_sse(bus)

    @router.get("/api/agents/{agent_id}/transcript")
    def agent_transcript(agent_id: str) -> TranscriptResponse:
        """The agent's conversation so far (its one session's history), so the
        chat UI can rebuild on load. Empty when the agent has never run."""
        agent = require_agent(deps.runs, agent_id)
        backend = get_backend(agent.backend)
        return TranscriptResponse(
            messages=backend.read_transcript(deps.settings, agent.session_id)
        )

    @router.get("/api/agents/{agent_id}/usage")
    def agent_usage(agent_id: str) -> Capability[UsageQuota]:
        """The account backing THIS agent's usage/quota (the rate-limit window),
        as its BACKEND reports it. ``supported: false`` when the backend has no
        such reader - distinct from a supported reader finding nothing. 404
        unknown."""
        return deps.diagnostics.usage(require_agent(deps.runs, agent_id))

    @router.get("/api/agents/{agent_id}/memory")
    def agent_memory(agent_id: str) -> Capability[MemoryFootprint]:
        """The agent's persistent on-disk footprint, as its BACKEND reports it.
        ``supported: false`` when the backend keeps nothing scufris can measure -
        not an all-zero footprint that reads as a measurement. 404 unknown."""
        return deps.diagnostics.memory(require_agent(deps.runs, agent_id))

    @router.get("/api/agents/{agent_id}/health")
    async def agent_health_endpoint(agent_id: str) -> AgentHealth:
        """Read-only diagnostics probed for THIS agent's backend (a claude agent
        probes the claude CLI, not codex). Resolves the orchestrator too, so its
        settings page shares this endpoint. 404 unknown; never raises otherwise.

        The MCP health rows are scoped to THIS agent's audience: the orchestrator
        gets its scufris + den servers, a sub-agent its callback server, a backend
        with no scufris MCP a single "none" row."""
        agent = await require_agent_async(deps.runs, agent_id)
        ensure_den_path(deps.settings)  # so the in-process den probe sees the den
        return await deps.diagnostics.health(agent)

    @router.get("/api/agents/{agent_id}/account")
    def agent_account(agent_id: str) -> AccountInfo:
        """The account backing THIS agent: its effective model, auth mode, and its
        backend's usage quota capability. 404 unknown."""
        return deps.diagnostics.account(require_agent(deps.runs, agent_id))

    @router.get("/api/agents/{agent_id}/tools")
    async def get_agent_scoped_tools(agent_id: str) -> Capability[list[AgentTool]]:
        """The scufris MCP tools THIS agent can actually call in its turns -
        AUDIENCE- and BACKEND-scoped, read-only. A codex or claude sub-agent gets
        only the ``agent`` callback server (request_input/report_back); the
        orchestrator gets its ``scufris`` + ``den`` servers; an agent whose backend
        does not wire the scufris MCP (opencode/mock, today) reports ``supported:
        false`` - it has no listing to give, which is not an empty one. This is what
        the agent's settings page shows, so the display matches what the agent
        really has - unlike the orchestrator-console ``/api/agent/tools``. 404
        unknown agent."""
        return await deps.diagnostics.tools(
            await require_agent_async(deps.runs, agent_id)
        )

    @router.get("/api/agents/{agent_id}/mcp")
    async def get_agent_scoped_mcp(agent_id: str) -> list[McpServerHealth]:
        """Live per-server health for THIS agent's audience: the orchestrator's
        ``scufris`` + ``den``, or a sub-agent's ``agent`` callback server. Empty when
        the agent's backend wires no scufris MCP (opencode/mock). 404 unknown
        agent."""
        agent = await require_agent_async(deps.runs, agent_id)
        ensure_den_path(deps.settings)
        return await deps.diagnostics.mcp(agent)

    return router


__all__ = [
    "AcknowledgeResult",
    "AgentChatRequest",
    "AgentForkRequest",
    "AgentReportBack",
    "AgentRequestInput",
    "AgentRunDeps",
    "AgentRunRequest",
    "AgentRunStatus",
    "CancelResult",
    "ReportBackResult",
    "RequestInputResult",
    "RunStarted",
    "build_agent_run_router",
    "drain_turn",
    "launch",
    "require_agent",
    "require_agent_async",
    "require_agent_project",
    "require_agent_project_async",
]
