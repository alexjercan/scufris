"""The singular `/api/agent/*` surface: the operator console's own agent.

Every route here is orchestrator-scoped. Most are compatibility aliases for the
plural `/api/agents/orchestrator/*` routes and answer out of the SAME services -
`AgentDiagnostics` for account/usage/memory/health, `AgentRunService` for the
record and the forked turn - which is the property that keeps the two surfaces
from drifting into two different answers for one question.

Three things live only here, because the console is their only caller: the
effective-config view and its whitelisted PATCH (the settings page), the
in-process "try it" tool runner, and the orchestrator's session switcher.

The session routes read through the orchestrator's BACKEND rather than a
provider disk scan, so codex/claude/opencode all work; the list itself comes
from the ownership registry, so a sub-agent's chat can never appear in the
operator's switcher.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from ..agent import AgentReply
from ..agent_diagnostics import (
    AccountInfo,
    AgentDiagnostics,
    mcp_servers_for_audience,
    probe_servers,
    tools_for_servers,
)
from ..agent_store import ORCHESTRATOR_ID, AgentStore
from ..backends import Capability, get_backend, session_info
from ..config import Settings, auth_mode_for_backend
from ..enums import AuthMode, Backend, PermissionMode
from ..env_bridge import ensure_den_path
from ..health import AgentHealth
from ..mcp_common import api_token_var
from ..mcp_models import AgentTool, McpServerHealth
from ..orchestrator import AgentRunService
from ..sessions import MemoryFootprint, SessionContext, SessionInfo, UsageQuota
from ..settings_store import SettingsReadOnly, SettingsStore, UnknownSettingKey
from ..supervisor import AgentSupervisor
from .agent_runs import drain_turn, launch, require_agent, require_agent_async
from .models import DeleteResult, TranscriptResponse


class AgentInfo(BaseModel):
    model: str
    # None for a backend with no login (mock); else the backend's auth mode.
    auth_mode: AuthMode | None
    enabled: bool


class ToolRunRequest(BaseModel):
    """Body for the "try it" runner: the args to pass the tool (name -> value)."""

    args: dict[str, object] = {}


class ToolRunResult(BaseModel):
    """The result of running one MCP tool: its text output and structured block."""

    ok: bool
    text: str
    structured: dict[str, object] = {}


class AgentConfig(BaseModel):
    """The agent's effective configuration, for the settings view.

    Seeded from environment variables and layered with persisted overrides.
    ``writable`` tells the UI whether config can be changed here; the codex
    sandbox is always ``read-only`` regardless.
    """

    enabled: bool
    backend: str
    model: str
    # None for a backend with no login (mock); else the backend's auth mode.
    auth_mode: AuthMode | None
    tools_enabled: bool
    sandbox: str
    # Whether this server accepts config writes (drives the UI: render controls
    # vs a read-only view). False when SCUFRIS_SETTINGS_WRITABLE is off.
    writable: bool


class AgentConfigUpdate(BaseModel):
    """A partial, whitelisted config update from the settings page.

    Every field is optional; only those present are applied. The whitelist is
    enforced by the store, but modelling the accepted keys here gives a typed
    request and rejects unknown keys at the API boundary.
    """

    model_config = ConfigDict(extra="forbid")

    agent_enabled: bool | None = None
    agent_backend: Backend | None = None
    agent_model: str | None = None
    claude_model: str | None = None
    agent_permission_mode: PermissionMode | None = None
    agent_tools_enabled: bool | None = None
    agent_timeout_seconds: float | None = None
    poll_seconds: float | None = None
    disabled_tools: list[str] | None = None
    # The scheduled host checks. Editable at runtime because the digest is tuned by
    # living with it: a threshold behind an env var and a restart never gets tuned.
    host_checks_enabled: bool | None = None
    host_watch_enabled: bool | None = None
    host_watch_interval_seconds: float | None = None
    host_digest_enabled: bool | None = None
    host_digest_at: str | None = None
    host_digest_muted_until: float | None = None
    check_disk_warn_percent: float | None = None
    check_disk_crit_percent: float | None = None
    check_temp_warn_celsius: float | None = None
    check_store_dead_paths: int | None = None
    check_flake_age_days: int | None = None
    check_escalate_gc: bool | None = None


class SessionsResponse(BaseModel):
    sessions: list[SessionInfo]
    current: str | None


class CurrentSession(BaseModel):
    current: str | None


class SessionAction(BaseModel):
    action: Literal["new", "switch"]
    session_id: str | None = None


class ForkRequest(BaseModel):
    source_id: str
    message_index: int
    text: str


class ForkResult(BaseModel):
    current: str | None
    reply: AgentReply


@dataclass(frozen=True)
class LegacyAgentDeps:
    """What the console's own agent surface reads.

    ``api_token`` is the machine token minted for this app. The "try it" runner
    needs it because a tool run IN THIS process may call this server's own gated
    API; an MCP subprocess gets the same token through its injected env. It is
    ``repr=False`` so the credential never renders in a traceback that prints
    these deps.
    """

    settings: Settings
    agents: AgentStore
    store: SettingsStore
    diagnostics: AgentDiagnostics
    runs: AgentRunService
    supervisor: AgentSupervisor
    api_token: str = field(repr=False)


def build_legacy_agent_router(deps: LegacyAgentDeps) -> APIRouter:
    """The operator console's orchestrator-scoped `/api/agent/*` surface."""
    router = APIRouter()

    @router.get("/api/agent/info")
    def get_agent_info() -> AgentInfo:
        """The model the agent drives, its auth mode, and whether it is enabled.

        Off the ORCHESTRATOR RECORD, not ``settings.agent_model`` - that key is
        the codex model slot, so a claude orchestrator used to report a codex
        model here while ``/api/agents/orchestrator/account`` reported the right
        one."""
        orchestrator = require_agent(deps.runs, ORCHESTRATOR_ID)
        return AgentInfo(
            model=orchestrator.model,
            auth_mode=auth_mode_for_backend(deps.settings, orchestrator.backend),
            enabled=deps.settings.agent_enabled,
        )

    def _agent_config() -> AgentConfig:
        """Build the effective-config view: live settings, but the model and auth
        mode off the orchestrator record (see ``/api/agent/info``)."""
        orchestrator = require_agent(deps.runs, ORCHESTRATOR_ID)
        return AgentConfig(
            enabled=deps.settings.agent_enabled,
            backend=orchestrator.backend,
            model=orchestrator.model,
            auth_mode=auth_mode_for_backend(deps.settings, orchestrator.backend),
            tools_enabled=deps.settings.agent_tools_enabled,
            sandbox="read-only",
            writable=deps.store.writable,
        )

    @router.get("/api/agent/config")
    def get_agent_config() -> AgentConfig:
        """The agent's effective configuration for the settings view."""
        return _agent_config()

    @router.patch("/api/agent/config")
    def patch_agent_config(update: AgentConfigUpdate) -> AgentConfig:
        """Apply a whitelisted config change; persist it; return effective config.

        403 when the server is read-only (SCUFRIS_SETTINGS_WRITABLE off), 422 for
        an unknown key or an invalid value. The change is live: it mutates the
        running settings and rebuilds the agent when enabled/backend change.
        """
        updates = update.model_dump(exclude_none=True)
        try:
            deps.store.apply(updates)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (UnknownSettingKey, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _agent_config()

    @router.get("/api/agent/tools")
    async def get_agent_tools() -> list[AgentTool]:
        """The full curated tool set for the operator console (the orchestrator's
        "try it" runner runs these IN-PROCESS, so this is the dashboard's own tool
        surface). Aggregates the orchestrator's two servers - ``scufris`` (agentic)
        and ``den`` (life) - each tool tagged with its server. For what a SPECIFIC
        agent can call in its own turns, see ``GET /api/agents/{id}/tools``.

        Deliberately scoped to the ORCHESTRATOR's audience, so the host agent's
        mutating propose tools are not here (they are on its ``host`` server; see
        ``GET /api/agents/host/tools``). The operator's own route to a host change
        is the approval queue over ``/api/host/actions``, which needs no tool
        runner - and a console that could propose would be a second, differently
        audited path to the same helper."""
        return await tools_for_servers(
            deps.settings, mcp_servers_for_audience(ORCHESTRATOR_ID)
        )

    @router.get("/api/agent/mcp")
    async def get_agent_mcp() -> list[McpServerHealth]:
        """Live per-server health for the operator console's "MCP tools" section:
        the orchestrator's ``scufris`` + ``den`` servers, each with a probe status
        (green/amber/red) and its tools carrying per-tool enabled/available flags.
        For a SPECIFIC agent's servers, see ``GET /api/agents/{id}/mcp``."""
        ensure_den_path(deps.settings)
        return await probe_servers(
            deps.settings, mcp_servers_for_audience(ORCHESTRATOR_ID)
        )

    @router.post("/api/agent/tools/{name}/run")
    async def run_agent_tool(name: str, req: ToolRunRequest) -> ToolRunResult:
        """Run ONE scufris MCP tool by name, in-process, bypassing codex/the agent.

        The operator console's "try it" runner: debug a single tool in isolation.
        Refuses a tool the operator disabled (403); an unknown tool is 404 and
        bad/missing/invalid args are 422 - never an uncontrolled 500. There is no
        gating setting: the tool set is already curated (fixed flags, bounded
        output, no arbitrary-command tool), and the UI adds a confirm step.

        Note: FastMCP wraps BOTH arg-validation errors and any exception raised
        inside a tool body as `ToolError`, so both map to 422 here. This is
        deliberate: the scufris tools return their errors as text rather than
        raising, so in practice the only `ToolError` that reaches this handler is
        the arg-validation case, for which 422 is the correct signal.
        """
        from mcp.server.fastmcp.exceptions import ToolError

        if name in set(deps.settings.disabled_tools):
            raise HTTPException(status_code=403, detail=f"tool {name!r} is disabled")
        # The console runs the orchestrator's tools, now spread across two servers
        # (scufris + den); find the one that owns this tool.
        target = None
        for _server_id, m in mcp_servers_for_audience(ORCHESTRATOR_ID):
            if any(t.name == name for t in await m.list_tools()):
                target = m
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {name!r}")
        # This runs the tool IN THIS process, so bridge the den path from settings
        # into the env the journal_* tools read (the agent path injects it into the
        # MCP subprocess instead; see env_bridge.ensure_den_path). No-op for
        # non-journal tools.
        ensure_den_path(deps.settings)
        # This tool may call THIS server's own API (`mcp_common._api_call`), which
        # is now gated. An MCP subprocess gets the machine token through its
        # injected env; an in-process run gets it here. A ContextVar rather than
        # os.environ so a second app in the same process cannot clobber it, and so
        # nothing else in the process picks it up ambiently; `asyncio.to_thread`
        # copies the context, so it survives the off-loop hop below.
        api_token_var.set(deps.api_token)
        try:
            # Run the tool OFF the event loop. FastMCP calls a SYNC tool inline
            # (`return fn(...)`), so an HTTP-backed tool's BLOCKING httpx call would
            # otherwise run on the loop - and when the base points at THIS server
            # (the common case now, see `ensure_api_base`), that self-loopback
            # request could never be served, hanging until timeout. A worker thread
            # with its own loop keeps the server loop free to answer the callback.
            raw = await asyncio.to_thread(
                lambda: asyncio.run(target.call_tool(name, req.args))
            )
        except ToolError as exc:
            # Bad/missing/invalid args (pydantic validation inside FastMCP).
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # FastMCP.call_tool returns (content_blocks, structured_dict) at runtime,
        # despite a looser type annotation; unpack defensively so a shape change
        # cannot 500 the endpoint.
        raw_any = cast(Any, raw)
        if isinstance(raw_any, tuple) and len(raw_any) == 2:
            blocks, structured = raw_any
        else:
            blocks, structured = raw_any, {}
        text = "".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"
        )
        return ToolRunResult(
            ok=True,
            text=text,
            structured=structured if isinstance(structured, dict) else {},
        )

    @router.get("/api/agent/health")
    async def get_agent_health() -> AgentHealth:
        """Read-only diagnostics for the operator console (never raises for the
        orchestrator, whose record is synthetic). The MCP rows are the
        orchestrator's scufris + den servers.

        Delegates to the same service as ``/api/agents/orchestrator/health``, so
        ``has_scufris_mcp`` follows the record's backend instead of defaulting to
        true for a backend that wires no scufris MCP."""
        orchestrator = await require_agent_async(deps.runs, ORCHESTRATOR_ID)
        ensure_den_path(deps.settings)  # so the in-process den probe sees the den
        return await deps.diagnostics.health(orchestrator)

    @router.get("/api/agent/sessions")
    def get_sessions() -> SessionsResponse:
        """List the orchestrator's own sessions (to switch between) + the current
        one. Driven by the ownership registry, not a provider disk scan: only
        sessions the registry attributes to the orchestrator appear, so a
        sub-agent's chat can never leak in (part 1). Each id is hydrated through
        the orchestrator's backend, so this works for codex/claude/opencode
        alike."""
        if not deps.settings.agent_enabled:
            return SessionsResponse(sessions=[], current=None)
        backend = get_backend(deps.agents.get(ORCHESTRATOR_ID).backend)
        infos = [
            info
            for sid in deps.agents.orchestrator_sessions()
            if (info := session_info(backend, deps.settings, sid)) is not None
        ]

        def _activity(info: SessionInfo) -> float:
            # Newest first by last activity; fall back to start time, then 0.
            when = info.updated_at or info.started_at
            return when.timestamp() if when is not None else 0.0

        infos.sort(key=_activity, reverse=True)
        return SessionsResponse(
            sessions=infos,
            current=deps.agents.orchestrator_session_id(),
        )

    @router.post("/api/agent/session")
    async def post_session(action: SessionAction) -> CurrentSession:
        """Start a new session or switch to an existing one for the next turn."""
        if not deps.settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        # Serialize on the orchestrator id (not the old "chat" key): its turns now
        # run through the supervisor keyed on that id, so a session switch cannot
        # interleave with an in-flight orchestrator turn.
        async with deps.supervisor.serialized(ORCHESTRATOR_ID):
            if action.action == "switch":
                if not action.session_id:
                    raise HTTPException(
                        status_code=422, detail="session_id required to switch"
                    )
                await asyncio.to_thread(
                    deps.agents.set_orchestrator_session, action.session_id
                )
            else:
                await asyncio.to_thread(deps.agents.set_orchestrator_session, None)
            return CurrentSession(
                current=await asyncio.to_thread(deps.agents.orchestrator_session_id)
            )

    @router.post("/api/agent/session/fork")
    async def fork_session(request: ForkRequest) -> ForkResult:
        """Fork a conversation: start a new session seeded with the turns before
        the edited message plus the edited text, and run it as the first turn.

        codex-exec has no native branch, so the prior turns are pasted as context.
        """
        if not deps.settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        orchestrator = await asyncio.to_thread(deps.agents.get, ORCHESTRATOR_ID)
        seed = await asyncio.to_thread(
            deps.runs.fork_seed,
            orchestrator,
            request.source_id,
            request.message_index,
            request.text,
        )
        # Drop the active session, then run the seed as a fresh turn. No outer
        # serialize() lock here: the launch already reserves the orchestrator's
        # serialize slot (and refuses a concurrent turn), so wrapping this in
        # supervisor.serialized(ORCHESTRATOR_ID) would self-deadlock on the same
        # key.
        #
        # What keeps a concurrent turn off the just-cleared session is that the
        # clear and the launch's slot claim are ONE synchronous step:
        # `AgentRunService.launch` runs its guard and claims the run registry
        # before its first await, and there is no await between it and the clear.
        # That is why the forked record is derived in memory rather than re-read -
        # a store round trip here would be exactly the gap a concurrent turn needs
        # to land on the cleared session and start a new chat instead of resuming
        # the operator's (review round 1, R1.3).
        forked = orchestrator.model_copy(update={"session_id": None})
        await asyncio.to_thread(deps.agents.set_orchestrator_session, None)
        _run_id, bus = await launch(deps.runs, forked, None, seed)
        done = await drain_turn(deps.runs, bus)
        return ForkResult(
            current=done.session_id
            or await asyncio.to_thread(deps.agents.orchestrator_session_id),
            reply=done.reply,
        )

    @router.get("/api/agent/context")
    def get_context() -> SessionContext | None:
        """The current session's context snapshot (window + token usage + counts),
        read through the orchestrator's backend so it works for codex/claude/
        opencode (codex keeps the rich token breakdown; others map read_status)."""
        if not deps.settings.agent_enabled:
            return None
        backend = get_backend(deps.agents.get(ORCHESTRATOR_ID).backend)
        return backend.read_context(
            deps.settings, deps.agents.orchestrator_session_id()
        )

    @router.get("/api/agent/session/{session_id}")
    def get_session_transcript(session_id: str) -> TranscriptResponse:
        """A session's past messages, so switching to it re-renders its history -
        read through the orchestrator's backend (codex/claude/opencode)."""
        if not deps.settings.agent_enabled:
            return TranscriptResponse(messages=[])
        backend = get_backend(deps.agents.get(ORCHESTRATOR_ID).backend)
        return TranscriptResponse(
            messages=backend.read_transcript(deps.settings, session_id)
        )

    @router.delete("/api/agent/session/{session_id}")
    async def delete_agent_session(session_id: str) -> DeleteResult:
        """Delete a session: remove its provider-side record via the orchestrator's
        backend (codex rollout / claude file / opencode daemon) AND forget it from
        the switcher history, so it leaves the list. ``forget`` also clears the
        current pointer when it was the active session; a backend with no provider
        delete still drops the session from the list."""
        if not deps.settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with deps.supervisor.serialized(ORCHESTRATOR_ID):
            orchestrator = await asyncio.to_thread(deps.agents.get, ORCHESTRATOR_ID)
            deleted = await get_backend(orchestrator.backend).delete_session(
                deps.settings, session_id
            )
            await asyncio.to_thread(deps.agents.forget_orchestrator_session, session_id)
            return DeleteResult(
                deleted=deleted,
                current=await asyncio.to_thread(deps.agents.orchestrator_session_id),
            )

    @router.get("/api/agent/usage")
    def get_usage() -> Capability[UsageQuota]:
        """Account-wide usage/quota (the weekly rate-limit window) for the
        orchestrator's backend. A compatibility alias for
        ``/api/agents/orchestrator/usage``, envelope included: ``supported:
        false`` when the backend has no usage reader."""
        return deps.diagnostics.usage(require_agent(deps.runs, ORCHESTRATOR_ID))

    @router.get("/api/agent/memory")
    def get_memory() -> Capability[MemoryFootprint]:
        """The orchestrator's persistent on-disk footprint, as its BACKEND reports
        it. A compatibility alias for ``/api/agents/orchestrator/memory``:
        ``supported: false`` beats an all-zero footprint that reads as a
        measurement."""
        return deps.diagnostics.memory(require_agent(deps.runs, ORCHESTRATOR_ID))

    @router.get("/api/agent/account")
    def get_account() -> AccountInfo:
        """The account backing the orchestrator: auth mode, model, and its
        backend's usage quota. A compatibility alias for
        ``/api/agents/orchestrator/account``."""
        return deps.diagnostics.account(require_agent(deps.runs, ORCHESTRATOR_ID))

    return router


__all__ = [
    "AgentConfig",
    "AgentConfigUpdate",
    "AgentInfo",
    "CurrentSession",
    "ForkRequest",
    "ForkResult",
    "LegacyAgentDeps",
    "SessionAction",
    "SessionsResponse",
    "ToolRunRequest",
    "ToolRunResult",
    "build_legacy_agent_router",
]
