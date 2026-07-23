"""The Scufris FastAPI application.

Serves a read-only JSON stats API and, when built, the static dashboard bundle.
The stats collector is injected so tests can supply a fake; production uses the
psutil-backed collector.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from . import sesh
from .agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
)
from .agent_store import (
    ORCHESTRATOR_ID,
    AgentNotFound,
    AgentRecord,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
)
from .backends import get_backend
from .config import (
    SERVER_ID_RE,
    McpServerSpec,
    Settings,
    auth_mode_for_backend,
    available_backends,
    backend_label,
    canonical_backend,
    default_model_for,
    models_for,
)
from .enums import AgentState, AuthMode, Backend, PermissionMode, RunPhase
from .eventbus import EventBus
from .health import AgentHealth, agent_health
from .logsetup import configure_logging, new_request_id, set_request_id
from .metrics import Collector, HostStats, PsutilCollector
from .processes import ProcessCollector, ProcessList, PsutilProcessCollector
from .projects import (
    DuplicateProject,
    InvalidProject,
    Project,
    ProjectNotFound,
    ProjectsReadOnly,
    ProjectStore,
    ProjectTask,
    read_project_tasks,
)
from .sessions import (
    MemoryFootprint,
    SessionContext,
    SessionInfo,
    TranscriptMessage,
    UsageQuota,
    delete_session,
    format_fork_seed,
    list_sessions,
    read_context,
    read_memory_footprint,
    read_transcript,
    read_usage,
    resolve_codex_home,
)
from .settings_store import (
    CannotDeleteProfile,
    DuplicateProfile,
    InvalidProfileName,
    SettingsReadOnly,
    SettingsStore,
    UnknownProfile,
    UnknownSettingKey,
)
from .supervisor import RunState, Supervisor

logger = logging.getLogger(__name__)


def _scufris_version() -> str:
    try:
        return metadata.version("scufris")
    except metadata.PackageNotFoundError:  # pragma: no cover - packaged has metadata
        return "0.0.0+unknown"


SCUFRIS_VERSION = _scufris_version()

# Shown at the top of /docs (Swagger) and /redoc. Markdown is rendered there.
API_DESCRIPTION = """\
The Scufris backend: a host dashboard and a multi-agent orchestrator.

It serves live host metrics, the main **orchestrator agent** chat (streamed over
SSE), first-class **projects**, and the **agents** that run on them - each agent
is bound to a project, driven by a swappable backend (codex or Claude Code), and
run as a supervised background job with live status and an event stream.

Endpoints are grouped by the tags below. Mutating endpoints under a writable
server are gated by `SCUFRIS_SETTINGS_WRITABLE`; agent turns run read-only unless
an agent has the per-agent write opt-in enabled.
"""

# Tag metadata drives the section ORDER and descriptions in /docs. Routes are
# assigned to these tags by path in `_route_tags` (below), so a single map keeps
# the grouping in one place rather than a `tags=` on every decorator.
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "host", "description": "Live host metrics: system stats and processes."},
    {
        "name": "app",
        "description": "Client-facing app configuration (poll interval, agent on/off).",
    },
    {
        "name": "chat",
        "description": "The main orchestrator agent chat - one turn (`/api/chat`) or streamed live over SSE (`/api/chat/stream`).",
    },
    {
        "name": "sessions",
        "description": "The chat agent's codex sessions: list, switch, fork, transcript, context window, usage/quota, on-disk memory and account.",
    },
    {
        "name": "settings",
        "description": "Agent configuration: effective config, MCP servers, named profiles, the tool catalog and health checks.",
    },
    {
        "name": "projects",
        "description": "First-class projects (a workspace an agent runs in) and their tatr tasks.",
    },
    {
        "name": "agents",
        "description": "The multi-agent orchestrator: agent records (CRUD) and running them - launch a goal, poll status, stream events.",
    },
]


def _route_tags(path: str) -> list[str]:
    """The OpenAPI tag for an API route, by path (see OPENAPI_TAGS).

    Order matters: the session/context family and the singular `/api/agent/...`
    settings family share a prefix, and the plural `/api/agents` must not be
    caught by the singular check.
    """
    if path in ("/api/stats", "/api/processes"):
        return ["host"]
    if path == "/api/config":
        return ["app"]
    if path.startswith("/api/chat") or path == "/api/agent/info":
        return ["chat"]
    if path.startswith("/api/agents"):
        return ["agents"]
    if path.startswith("/api/projects"):
        return ["projects"]
    session_paths = (
        "/api/agent/sessions",
        "/api/agent/session",
        "/api/agent/context",
        "/api/agent/usage",
        "/api/agent/memory",
        "/api/agent/account",
    )
    if any(path == p or path.startswith(p + "/") for p in session_paths):
        return ["sessions"]
    if path.startswith("/api/agent/"):
        return ["settings"]
    return []


class _NoCacheStaticFiles(StaticFiles):
    """Serve the SPA bundle with `Cache-Control: no-cache`.

    The bundle filenames are not content-hashed (`agent.js`, `index.html`), so
    without this a browser applies heuristic freshness and can keep running a
    stale bundle for hours without revalidating. `no-cache` forces revalidation
    on every load - the ETag still yields a fast 304 when unchanged, but a
    rebuilt bundle is picked up immediately.
    """

    async def get_response(self, path: str, scope: object) -> Response:
        response = await super().get_response(path, scope)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = "no-cache"
        return response


class AppConfig(BaseModel):
    poll_seconds: float
    agent_enabled: bool


class AgentInfo(BaseModel):
    model: str
    # None for a backend with no login (mock); else the backend's auth mode.
    auth_mode: AuthMode | None
    enabled: bool


class AccountInfo(BaseModel):
    """The account backing the agent, for the console's Account panel."""

    # None for a backend with no login (mock); else the backend's auth mode
    # (codex -> chatgpt/api_key, claude -> claude_ai/api_key).
    auth_mode: AuthMode | None
    model: str
    enabled: bool
    quota: UsageQuota | None = None


class ToolParam(BaseModel):
    """One input parameter of an MCP tool, distilled from its JSON input schema.

    The "try it" runner (settings page) generates a form field from each param:
    ``type`` picks the input kind (text/number/checkbox), ``required`` marks it.
    """

    name: str
    type: str = "string"  # JSON-schema type: string/integer/number/boolean/...
    required: bool = False
    description: str = ""
    default: object | None = None


class AgentTool(BaseModel):
    name: str
    description: str
    server: str = "scufris"  # the MCP server that exposes it
    args: list[str] = []  # parameter names, from the tool's input schema
    parameters: list[ToolParam] = []  # full param schema, for the "try it" runner
    enabled: bool = True  # False when the operator disabled it (disabled_tools)


def _tool_parameters(input_schema: object) -> list[ToolParam]:
    """Distill a tool's JSON ``inputSchema`` into typed params for the runner.

    Reads ``properties`` (name -> {type, description, default}) and the top-level
    ``required`` list. Unknown/missing types fall back to "string" so the form
    still renders a text input. Best-effort: a malformed schema yields [].
    """
    if not isinstance(input_schema, dict):
        return []
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return []
    required = input_schema.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    params: list[ToolParam] = []
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        raw_type = spec.get("type")
        params.append(
            ToolParam(
                name=str(name),
                type=raw_type if isinstance(raw_type, str) else "string",
                required=name in required_set,
                description=str(spec.get("description") or ""),
                default=spec.get("default"),
            )
        )
    return params


class ToolRunRequest(BaseModel):
    """Body for the "try it" runner: the args to pass the tool (name -> value)."""

    args: dict[str, object] = {}


class ToolRunResult(BaseModel):
    """The result of running one MCP tool: its text output and structured block."""

    ok: bool
    text: str
    structured: dict[str, object] = {}


class McpServerInfo(BaseModel):
    """One MCP server the agent has registered, for the read-only settings view."""

    id: str
    source: str  # "built-in" | "configured"


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
    mcp_servers: list[McpServerInfo]
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
    mcp_servers: list[McpServerSpec] | None = None
    disabled_tools: list[str] | None = None


class ProfilesResponse(BaseModel):
    profiles: list[str]
    active: str


class ProfileCreate(BaseModel):
    name: str
    # Seed the new profile from the active one's overrides (vs an empty profile).
    copy_from_active: bool = True


class ProfileActivate(BaseModel):
    name: str


class ProjectCreate(BaseModel):
    name: str
    cwd: str
    language: str = ""
    description: str = ""


class ProjectNew(BaseModel):
    """Create a BRAND-NEW project directory under one of the base dirs, then
    register it. `base` must be one of `project_base_dirs` (the endpoint mkdirs
    under it); registering an already-existing dir uses `POST /api/projects`."""

    name: str
    base: str


class DiscoveredProject(BaseModel):
    """A candidate project directory for the Projects page: a discovered dir, a
    registered project, or both. `registered`/`project_id` mark the ones already
    tracked so the UI can offer register vs open."""

    path: str
    name: str
    language: str = ""
    registered: bool = False
    project_id: str | None = None


class DiscoveredProjects(BaseModel):
    """The Projects page payload: the discovered-union-registered directories plus
    the base dirs offered in the create form's picker."""

    projects: list[DiscoveredProject]
    base_dirs: list[str]


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    cwd: str | None = None
    language: str | None = None
    description: str | None = None


class AgentCreate(BaseModel):
    name: str
    project_id: str
    backend: str | None = None
    model: str | None = None
    description: str = ""
    goal: str = ""
    task_id: str = ""
    permission_mode: PermissionMode = PermissionMode.MANUAL


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    backend: str | None = None
    model: str | None = None
    description: str | None = None
    goal: str | None = None
    task_id: str | None = None
    permission_mode: PermissionMode | None = None


class BackendOption(BaseModel):
    # One selectable backend for the agent create/settings pickers: its id, a
    # friendly label, the default model stamped when it is chosen, and the
    # suggested model catalog (autocomplete; the field still accepts free text).
    id: str
    label: str
    default_model: str
    models: list[str]


class AgentRunRequest(BaseModel):
    # An optional goal override; falls back to the agent's stored goal.
    goal: str | None = None


class AgentChatRequest(BaseModel):
    # One user turn of a per-agent conversation.
    message: str


class AgentForkRequest(BaseModel):
    # Revert-fork a single-session agent: rewind its one session to
    # ``message_index`` and continue from the edited ``text``.
    message_index: int
    text: str


class RunStarted(BaseModel):
    agent_id: str
    state: str


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


class SessionsResponse(BaseModel):
    sessions: list[SessionInfo]
    current: str | None


class CurrentSession(BaseModel):
    current: str | None


class TranscriptResponse(BaseModel):
    messages: list[TranscriptMessage]


class DeleteResult(BaseModel):
    deleted: bool
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


class ImageAttachment(BaseModel):
    """One image attached to a chat turn (base64 payload + its MIME type)."""

    data_base64: str
    mime: str


class ChatRequest(BaseModel):
    message: str
    image: ImageAttachment | None = None


# Reject oversized uploads (decoded) so a bad/huge payload cannot exhaust memory.
_MAX_IMAGE_BYTES = 12 * 1024 * 1024


def _write_image_to_temp(image: ImageAttachment) -> tuple[str, str]:
    """Decode a base64 image attachment to a temp file for codex to read.

    Returns ``(tmpdir, path)`` (the caller removes ``tmpdir`` after the turn).
    Raises ``ValueError`` on a non-image type, invalid base64, or oversize payload.
    """
    if not image.mime.startswith("image/"):
        raise ValueError(f"unsupported attachment type: {image.mime}")
    try:
        data = base64.b64decode(image.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("attachment is not valid base64") from exc
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("attachment is too large")
    tmpdir = tempfile.mkdtemp(prefix="scufris-img-")
    ext = mimetypes.guess_extension(image.mime) or ".png"
    path = Path(tmpdir) / f"attachment{ext}"
    path.write_bytes(data)
    return tmpdir, str(path)


def _validate_mcp_spec(spec: McpServerSpec) -> None:
    """Reject a bad MCP server id/command at the API boundary (422)."""
    if spec.id == "scufris" or not re.fullmatch(SERVER_ID_RE, spec.id):
        raise HTTPException(
            status_code=422, detail=f"invalid MCP server id: {spec.id!r}"
        )
    if not spec.command.strip():
        raise HTTPException(
            status_code=422, detail="MCP server command must not be empty"
        )


def create_app(
    collector: Collector | None = None,
    settings: Settings | None = None,
    process_collector: ProcessCollector | None = None,
) -> FastAPI:
    settings = settings or Settings()
    collector = collector or PsutilCollector()
    process_collector = process_collector or PsutilProcessCollector()
    projects = ProjectStore(settings)
    # First-class agents: named, project-bound records (A1). Running one is A3.
    # The landing orchestrator is a reserved record in this store (B5bc), so the
    # landing chat + session endpoints run through the same backend path as any
    # other agent - there is no longer a separate injected `Agent` object.
    agents = AgentStore(settings, projects)

    # Runtime-mutable settings: env base with persisted overrides layered on.
    # Mutations happen in place, so the closures below read the new value live
    # (the backend is resolved per turn via get_backend(agent.backend), not
    # cached). Switching the orchestrator's backend drops its active session so a
    # stale cross-backend session id is never resumed under the new backend.
    def _on_settings_change(changed: set[str]) -> None:
        if "agent_backend" in changed:
            agents.set_orchestrator_session(None)

    store = SettingsStore(settings, on_change=_on_settings_change)
    # Agent turns run as background jobs under the supervisor (ADR-001), not
    # inside the request. A dropped client no longer cancels a turn, and there is
    # no request timeout - a per-run heartbeat guards a genuinely stalled turn.
    supervisor = Supervisor(max_concurrent=settings.agent_max_concurrent)
    # The latest supervisor run id for each agent (a run id is unique per launch;
    # the agent id serializes them). Lets the status/events endpoints find an
    # agent's current run without colliding on re-runs of the same agent.
    agent_runs: dict[str, str] = {}
    # Codex sessions are not concurrency-safe, so an agent's turns run one at a
    # time: `_launch_agent_turn` reserves the supervisor's serialize slot keyed on
    # `agent.id`. The orchestrator's session-mutating endpoints (reset/new/switch/
    # delete) reserve the SAME key via `supervisor.serialized(ORCHESTRATOR_ID)`, so
    # they cannot interleave with an in-flight orchestrator turn - and because a
    # turn reserves its slot synchronously in `start()`, a mutation arriving right
    # after cannot slip in front of its own turn. (fork is the exception: it
    # LAUNCHES a turn, so it must NOT hold the lock or it self-deadlocks on the
    # key `_launch_agent_turn` reserves - see fork_session.)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await supervisor.aclose()  # cancel any in-flight runs on shutdown

    app = FastAPI(
        title="Scufris API",
        summary="Scuffed Jarvis: a host dashboard and multi-agent orchestrator.",
        description=API_DESCRIPTION,
        version=SCUFRIS_VERSION,
        lifespan=_lifespan,
        openapi_tags=OPENAPI_TAGS,
    )
    # Exposed for tests and future per-agent endpoints (A3/A4).
    app.state.supervisor = supervisor
    # Exposed so tests can seed the orchestrator's active session directly (the
    # landing session state now lives in the store, not an injected agent).
    app.state.agents = agents

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Tag each request with an id and log method/path/status/duration.

        At DEBUG so `--debug` shows every request without the default INFO being
        flooded by the dashboard's 2s stats/processes polling; 5xx at WARNING.
        """
        set_request_id(new_request_id())
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        level = logging.WARNING if response.status_code >= 500 else logging.DEBUG
        logger.log(
            level,
            "%s %s -> %d in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/api/stats")
    def get_stats() -> HostStats:
        """Return a fresh read-only snapshot of host metrics."""
        return collector.sample()

    @app.get("/api/processes")
    def get_processes() -> ProcessList:
        """Return current processes aggregated by application."""
        return process_collector.sample()

    @app.get("/api/config")
    def get_config() -> AppConfig:
        """Client-facing knobs: poll interval and whether the agent is on."""
        return AppConfig(
            poll_seconds=settings.poll_seconds, agent_enabled=settings.agent_enabled
        )

    @app.get("/api/agent/info")
    def get_agent_info() -> AgentInfo:
        """The model the agent drives, its auth mode, and whether it is enabled."""
        return AgentInfo(
            model=settings.agent_model,
            auth_mode=auth_mode_for_backend(settings, settings.agent_backend),
            enabled=settings.agent_enabled,
        )

    def _agent_config() -> AgentConfig:
        """Build the effective-config view from the live settings."""
        servers: list[McpServerInfo] = []
        if settings.agent_tools_enabled:
            servers.append(McpServerInfo(id="scufris", source="built-in"))
        servers += [
            McpServerInfo(id=spec.id, source="configured")
            for spec in settings.mcp_servers
        ]
        return AgentConfig(
            enabled=settings.agent_enabled,
            backend=settings.agent_backend,
            model=settings.agent_model,
            auth_mode=auth_mode_for_backend(settings, settings.agent_backend),
            tools_enabled=settings.agent_tools_enabled,
            sandbox="read-only",
            mcp_servers=servers,
            writable=store.writable,
        )

    @app.get("/api/agent/config")
    def get_agent_config() -> AgentConfig:
        """The agent's effective configuration for the settings view."""
        return _agent_config()

    @app.patch("/api/agent/config")
    def patch_agent_config(update: AgentConfigUpdate) -> AgentConfig:
        """Apply a whitelisted config change; persist it; return effective config.

        403 when the server is read-only (SCUFRIS_SETTINGS_WRITABLE off), 422 for
        an unknown key or an invalid value. The change is live: it mutates the
        running settings and rebuilds the agent when enabled/backend change.
        """
        # Reject a bad MCP server id at the boundary so a user add fails loudly
        # (env-declared servers are skipped silently by the agent instead).
        for spec in update.mcp_servers or []:
            _validate_mcp_spec(spec)
        updates = update.model_dump(exclude_none=True)
        try:
            store.apply(updates)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (UnknownSettingKey, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _agent_config()

    def _apply_mcp_servers(servers: list[McpServerSpec]) -> None:
        try:
            store.apply({"mcp_servers": servers})
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (UnknownSettingKey, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/mcp_servers")
    def add_mcp_server(spec: McpServerSpec) -> AgentConfig:
        """Append one MCP server. Incremental so the client need not resend the
        whole list (it does not know each server's command/args)."""
        _validate_mcp_spec(spec)
        if any(s.id == spec.id for s in settings.mcp_servers):
            raise HTTPException(
                status_code=409, detail=f"MCP server {spec.id!r} already exists"
            )
        _apply_mcp_servers([*settings.mcp_servers, spec])
        return _agent_config()

    @app.delete("/api/agent/mcp_servers/{server_id}")
    def remove_mcp_server(server_id: str) -> AgentConfig:
        """Remove an MCP server by id (404 if absent)."""
        remaining = [s for s in settings.mcp_servers if s.id != server_id]
        if len(remaining) == len(settings.mcp_servers):
            raise HTTPException(status_code=404, detail=f"no MCP server {server_id!r}")
        _apply_mcp_servers(remaining)
        return _agent_config()

    def _profiles() -> ProfilesResponse:
        return ProfilesResponse(
            profiles=store.profile_names(), active=store.active_profile
        )

    @app.get("/api/agent/profiles")
    def get_profiles() -> ProfilesResponse:
        """Named config profiles and which one is active."""
        return _profiles()

    @app.post("/api/agent/profiles")
    def create_profile(req: ProfileCreate) -> ProfilesResponse:
        """Create a profile (copying the active one's overrides by default)."""
        try:
            store.create_profile(req.name, copy_from_active=req.copy_from_active)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except DuplicateProfile as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidProfileName as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _profiles()

    @app.post("/api/agent/profiles/activate")
    def activate_profile(req: ProfileActivate) -> AgentConfig:
        """Switch the active profile; return the new effective config."""
        try:
            store.activate(req.name)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except UnknownProfile as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _agent_config()

    @app.delete("/api/agent/profiles/{name}")
    def delete_profile(name: str) -> ProfilesResponse:
        """Delete a profile; refuses the active or the last remaining one."""
        try:
            store.delete_profile(name)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except UnknownProfile as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CannotDeleteProfile as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _profiles()

    @app.get("/api/projects")
    def list_projects() -> list[Project]:
        """All projects, sorted by name."""
        return projects.list()

    @app.post("/api/projects")
    def create_project(req: ProjectCreate) -> Project:
        """Create a project; 422 for a bad name/cwd, 403 read-only."""
        try:
            return projects.create(
                name=req.name,
                cwd=req.cwd,
                language=req.language,
                description=req.description,
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (InvalidProject, DuplicateProject) as exc:
            code = 409 if isinstance(exc, DuplicateProject) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.get("/api/projects/discovered")
    def list_discovered_projects() -> DiscoveredProjects:
        """Directories discovered under the base dirs UNION the registered
        projects, each flagged with whether it is already registered, plus the
        base dirs for the create form's picker - the Projects page's source of
        truth. Declared before `/api/projects/{id}` so "discovered" is not parsed
        as a project id."""
        by_path: dict[str, DiscoveredProject] = {}
        for cand in sesh.discover(settings.project_base_dirs):
            by_path[cand.path] = DiscoveredProject(
                path=cand.path, name=cand.name, language=cand.language
            )
        # Mark discovered dirs that are registered, and ADD registered projects
        # whose cwd is not among the discovered dirs (registered outside a base).
        for project in projects.list():
            key = str(Path(project.cwd).resolve())
            existing = by_path.get(key)
            if existing is not None:
                existing.registered = True
                existing.project_id = project.id
            else:
                by_path[key] = DiscoveredProject(
                    path=key,
                    name=project.name,
                    language=project.language,
                    registered=True,
                    project_id=project.id,
                )
        ordered = sorted(by_path.values(), key=lambda d: (d.name.lower(), d.path))
        base_dirs = [str(b.expanduser()) for b in settings.project_base_dirs]
        return DiscoveredProjects(projects=ordered, base_dirs=base_dirs)

    @app.post("/api/projects/new")
    def create_new_project(req: ProjectNew) -> Project:
        """Make a NEW project directory under an allowed base dir and register it.
        422 for a base outside `project_base_dirs` or an unsafe name, 409 on an id
        collision, 403 read-only."""
        # Guard writability BEFORE the mkdir so a read-only server never has a
        # directory created as a side effect of a refused request.
        if not projects.writable:
            raise HTTPException(
                status_code=403, detail="projects are read-only on this server"
            )
        allowed = {
            str(base.expanduser().resolve()): base.expanduser()
            for base in settings.project_base_dirs
        }
        chosen = allowed.get(str(Path(req.base).expanduser().resolve()))
        if chosen is None:
            raise HTTPException(
                status_code=422,
                detail="base must be one of the configured project base dirs",
            )
        try:
            path = sesh.create(req.name, chosen)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return projects.create(
                name=req.name,
                cwd=str(path),
                language=sesh.infer_language(path),
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (InvalidProject, DuplicateProject) as exc:
            code = 409 if isinstance(exc, DuplicateProject) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> Project:
        """One project by id; 404 if unknown."""
        try:
            return projects.get(project_id)
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, req: ProjectUpdate) -> Project:
        """Update a project's fields; 404 unknown, 422 invalid, 403 read-only."""
        try:
            return projects.update(
                project_id,
                name=req.name,
                cwd=req.cwd,
                language=req.language,
                description=req.description,
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        except InvalidProject as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> DeleteResult:
        """Delete a project; 404 unknown, 403 read-only."""
        try:
            projects.delete(project_id)
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        return DeleteResult(deleted=True, current=None)

    @app.get("/api/projects/{project_id}/tasks")
    def get_project_tasks(project_id: str) -> list[ProjectTask]:
        """The project's tatr tasks (its specs); empty when it has no tasks/."""
        try:
            project = projects.get(project_id)
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        return read_project_tasks(project.cwd)

    @app.get("/api/agents")
    def list_agents() -> list[AgentRecord]:
        """All configured agents, sorted by name."""
        return agents.list()

    @app.post("/api/agents")
    def create_agent(req: AgentCreate) -> AgentRecord:
        """Create an agent bound to a project; 422 bad field/unknown project,
        403 read-only."""
        try:
            return agents.create(
                name=req.name,
                project_id=req.project_id,
                backend=req.backend,
                model=req.model,
                description=req.description,
                goal=req.goal,
                task_id=req.task_id,
                permission_mode=req.permission_mode,
            )
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except InvalidAgent as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agents/backends")
    def list_agent_backends() -> list[BackendOption]:
        """The backends an agent may use (mock only when the dev flag is on),
        each with its friendly label and default model, so the create/settings
        pickers are server-authoritative. Declared before /api/agents/{id} so
        "backends" is not parsed as an agent id."""
        return [
            BackendOption(
                id=b,
                label=backend_label(b),
                default_model=default_model_for(settings, b),
                models=models_for(settings, b),
            )
            for b in available_backends(settings)
        ]

    @app.get("/api/agents/{agent_id}")
    def get_agent(agent_id: str) -> AgentRecord:
        """One agent by id; 404 if unknown."""
        try:
            return agents.get(agent_id)
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc

    @app.patch("/api/agents/{agent_id}")
    def update_agent(agent_id: str, req: AgentUpdate) -> AgentRecord:
        """Update an agent's config; 404 unknown, 422 invalid, 403 read-only.

        The orchestrator has no agents.json row - its config lives in the settings
        store - so its edits (backend/model/permission_mode) route THERE and it
        reads them back through the synthetic record. Every other agent updates its
        own record. The unified settings form (U3) is identical either way."""
        if agent_id == ORCHESTRATOR_ID:
            return _update_orchestrator(req)
        try:
            return agents.update(
                agent_id,
                name=req.name,
                backend=req.backend,
                model=req.model,
                description=req.description,
                goal=req.goal,
                task_id=req.task_id,
                permission_mode=req.permission_mode,
            )
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReservedAgent as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc
        except InvalidAgent as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _update_orchestrator(req: AgentUpdate) -> AgentRecord:
        """Apply the orchestrator's editable fields to the SETTINGS store, then
        return the refreshed synthetic record. Name/description/goal/task_id are
        fixed for the orchestrator and ignored. Model follows the EFFECTIVE backend
        (codex -> agent_model, claude -> claude_model, opencode -> opencode_model);
        a blank model re-defaults.
        A backend change clears its session via the store's on_change wiring."""
        updates: dict[str, object] = {}
        eff_backend = canonical_backend(
            req.backend if req.backend is not None else settings.agent_backend
        )
        if req.backend is not None:
            updates["agent_backend"] = req.backend
        if req.model is not None:
            model = req.model.strip() or default_model_for(settings, eff_backend)
            key = {
                "claude": "claude_model",
                "opencode": "opencode_model",
            }.get(eff_backend, "agent_model")
            updates[key] = model
        if req.permission_mode is not None:
            updates["agent_permission_mode"] = req.permission_mode
        if updates:
            try:
                store.apply(updates)
            except SettingsReadOnly as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (UnknownSettingKey, ValidationError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return agents.get(ORCHESTRATOR_ID)

    @app.delete("/api/agents/{agent_id}")
    def delete_agent(agent_id: str) -> DeleteResult:
        """Delete an agent; 404 unknown, 403 read-only or reserved."""
        try:
            agents.delete(agent_id)
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReservedAgent as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return DeleteResult(deleted=True, current=None)

    def _require_agent(agent_id: str) -> AgentRecord:
        try:
            return agents.get(agent_id)
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc

    def _require_agent_project(agent: AgentRecord) -> Project | None:
        # The reserved orchestrator (and only it) has no project binding: it runs
        # in the server cwd. Everyone else must resolve to a real project.
        if not agent.project_id:
            return None
        try:
            return projects.get(agent.project_id)
        except ProjectNotFound as exc:
            raise HTTPException(
                status_code=422, detail="agent's project no longer exists"
            ) from exc

    def _launch_agent_turn(
        agent: AgentRecord,
        project: Project | None,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        on_done: Callable[[], None] | None = None,
    ) -> tuple[str, EventBus]:
        """Stream one turn of ``prompt`` through the agent's backend (resuming its
        session), on the SAME supervisor + event bus + agent-run registry as a
        goal run. Persists the (possibly new) session id and terminal state.
        Shared by ``run`` (goal), per-agent ``chat`` (message), and the landing
        orchestrator chat (B5bc). Raises HTTPException 409 when a run/chat for
        this agent is already active."""
        prev_run = agent_runs.get(agent.id)
        prev_state = supervisor.status(prev_run) if prev_run else None
        if prev_state is not None and prev_state.state in ("queued", "running"):
            raise HTTPException(
                status_code=409, detail="a run for this agent is already active"
            )

        backend = get_backend(agent.backend)
        captured: dict[str, str] = {}

        async def turn_stream() -> AsyncIterator[StreamEvent]:
            async for event in backend.stream(
                settings,
                prompt,
                session_id=agent.session_id,
                cwd=project.cwd if project is not None else None,
                image_paths=image_paths,
                permission_mode=agent.permission_mode,
                is_orchestrator=agent.id == ORCHESTRATOR_ID,
            ):
                if isinstance(event, StreamDone):
                    if event.session_id:
                        captured["session_id"] = event.session_id
                    # The final reply text is the durable outcome's message (BC1),
                    # so the orchestrator can read what a finished agent said/asked
                    # after the per-run bus has closed.
                    captured["message"] = event.reply.text
                yield event

        run_id = f"{agent.id}:{uuid.uuid4().hex}"

        def persist(run_state: RunState) -> None:
            # Best-effort: if the agent was deleted mid-run, mark_finished raises
            # AgentNotFound, which the supervisor swallows and logs - the terminal
            # state just is not persisted for a record that no longer exists.
            agents.mark_finished(
                agent.id,
                state=(
                    AgentState.DONE
                    if run_state.state == RunPhase.DONE
                    else AgentState.ERROR
                ),
                session_id=captured.get("session_id"),
                # Key the session under the backend this turn RAN on (the
                # launch-time snapshot), not whatever the current config is - a
                # backend switch that raced the turn must not mislabel it.
                backend=agent.backend,
                message=captured.get("message", ""),
                run_id=run_id,
            )
            # Turn-owned cleanup (e.g. an attached image tempdir) runs when the
            # run ends, not when a relay disconnects.
            if on_done is not None:
                on_done()

        agent_runs[agent.id] = run_id
        agents.mark_running(agent.id)
        bus = supervisor.start(
            run_id,
            turn_stream,
            serialize_key=agent.id,
            budget_seconds=None,
            heartbeat_seconds=settings.agent_heartbeat_seconds,
            on_complete=persist,
        )
        return run_id, bus

    async def _drain_turn(bus: EventBus) -> StreamDone:
        """Consume a background turn's event bus and return its terminal
        ``StreamDone`` (reply + session id). Used by the non-streaming landing
        chat and fork, which need the whole reply rather than an SSE relay.
        Raises 503 on a ``StreamError`` and 500 if the turn ends without a
        terminal event. Reading the session id off the done event (not the store)
        avoids racing the on_complete persist callback."""
        async for _seq, event in bus.subscribe(after_seq=0):
            if isinstance(event, StreamDone):
                return event
            if isinstance(event, StreamError):
                raise HTTPException(status_code=503, detail=event.detail)
        raise HTTPException(status_code=500, detail="turn ended without a reply")

    @app.post("/api/agents/{agent_id}/run")
    async def run_agent(agent_id: str, req: AgentRunRequest) -> RunStarted:
        """Launch a supervised background run for the agent, scoped to its project
        cwd via its configured backend. 404 unknown, 422 no goal / missing project,
        409 a run is already active.

        Async so it runs on the event loop thread - the supervisor schedules the
        background run via ``asyncio.create_task``, which needs a running loop (a
        sync endpoint runs in a worker thread with none)."""
        agent = _require_agent(agent_id)
        goal = (req.goal if req.goal is not None else agent.goal).strip()
        if not goal:
            raise HTTPException(
                status_code=422, detail="agent has no goal; provide one to run"
            )
        project = _require_agent_project(agent)
        run_id, _bus = _launch_agent_turn(agent, project, goal)
        # Report the supervisor's actual state (usually "queued" until a slot is
        # free), not an assumed "running".
        started = supervisor.status(run_id)
        return RunStarted(
            agent_id=agent_id, state=started.state if started is not None else "running"
        )

    @app.get("/api/agents/{agent_id}/status")
    def agent_run_status(agent_id: str) -> AgentRunStatus:
        """Merge the live Supervisor run-state with the backend's read-only
        rollout/session progress for the agent."""
        agent = _require_agent(agent_id)
        run_id = agent_runs.get(agent_id)
        run_state = supervisor.status(run_id) if run_id else None
        state = run_state.state if run_state is not None else agent.state
        backend = get_backend(agent.backend)
        progress = backend.read_status(settings, agent.session_id)
        result = AgentRunStatus(
            agent_id=agent_id, state=state, session_id=agent.session_id
        )
        if progress is not None:
            result.turns = progress.turns
            result.tool_calls = progress.tool_calls
            result.input_tokens = progress.input_tokens
            result.output_tokens = progress.output_tokens
            result.context_window = progress.context_window
            result.last_message = progress.last_message
            result.updated_at = progress.updated_at
        return result

    def _relay_bus_sse(bus: EventBus, after_seq: int = 0) -> StreamingResponse:
        """Relay an event bus as an SSE response (replay events after ``after_seq``,
        then live). Shared by the agent events + chat endpoints."""

        async def events() -> AsyncIterator[str]:
            yield f":{' ' * 2048}\n\n"
            async for seq, event in bus.subscribe(after_seq=after_seq):
                yield f"id: {seq}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/agents/{agent_id}/events")
    async def agent_events(agent_id: str, http_request: Request) -> StreamingResponse:
        """Relay the agent's current run event bus as SSE (drop-safe; a reconnect
        replays via Last-Event-ID). 404 when the agent has no live run bus."""
        _require_agent(agent_id)
        run_id = agent_runs.get(agent_id)
        bus = supervisor.bus(run_id) if run_id else None
        if bus is None:
            raise HTTPException(status_code=404, detail="no active run for this agent")

        after_seq = 0
        last_event_id = http_request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            after_seq = int(last_event_id)
        return _relay_bus_sse(bus, after_seq)

    @app.post("/api/agents/{agent_id}/chat")
    async def agent_chat(agent_id: str, req: AgentChatRequest) -> StreamingResponse:
        """Stream one chat turn with the agent as SSE, resuming its one session.
        Runs over the SAME supervisor + event bus + agent-run registry as a goal
        run, so ``/status`` and ``/events`` reflect the turn and a concurrent
        turn is refused (409). 404 unknown agent, 422 empty message / missing
        project. The persist callback writes the (possibly new) session id back,
        so the next turn resumes it."""
        agent = _require_agent(agent_id)
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        project = _require_agent_project(agent)
        _run_id, bus = _launch_agent_turn(agent, project, message)
        return _relay_bus_sse(bus)

    @app.post("/api/agents/{agent_id}/fork")
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
        agent = _require_agent(agent_id)
        if agent.id == ORCHESTRATOR_ID:
            raise HTTPException(
                status_code=409,
                detail="the orchestrator forks via /api/agent/session/fork",
            )
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="message must not be empty")
        project = _require_agent_project(agent)
        backend = get_backend(agent.backend)
        messages = backend.read_transcript(settings, agent.session_id)
        cut = max(0, req.message_index)
        seed = format_fork_seed(messages[:cut], text)
        # Launch against a session-cleared copy so the seed opens a fresh session
        # (the revert). The turn still runs under the real agent id, so the persist
        # callback writes the new session id back to the actual record.
        reverted = agent.model_copy(update={"session_id": None})
        _run_id, bus = _launch_agent_turn(reverted, project, seed)
        return _relay_bus_sse(bus)

    @app.get("/api/agents/{agent_id}/transcript")
    def agent_transcript(agent_id: str) -> TranscriptResponse:
        """The agent's conversation so far (its one session's history), so the
        chat UI can rebuild on load. Empty when the agent has never run."""
        agent = _require_agent(agent_id)
        backend = get_backend(agent.backend)
        return TranscriptResponse(
            messages=backend.read_transcript(settings, agent.session_id)
        )

    def _agent_is_codex(agent: AgentRecord) -> bool:
        # usage/memory/account are codex-account-level (per codex_home); claude has
        # no rollout-usage reader in scufris, so a non-codex agent's panels are
        # None/empty. Dispatch lives here so the three endpoints stay one-liners.
        return canonical_backend(agent.backend) == "codex"

    @app.get("/api/agents/{agent_id}/usage")
    def agent_usage(agent_id: str) -> UsageQuota | None:
        """The account backing THIS agent's usage/quota (the rate-limit window).
        None for a non-codex agent (no equivalent reader). 404 unknown."""
        agent = _require_agent(agent_id)
        if not _agent_is_codex(agent):
            return None
        return read_usage(resolve_codex_home(settings))

    @app.get("/api/agents/{agent_id}/memory")
    def agent_memory(agent_id: str) -> MemoryFootprint:
        """The agent's persistent on-disk footprint (codex rollouts). An empty
        footprint for a non-codex agent. 404 unknown."""
        agent = _require_agent(agent_id)
        if not _agent_is_codex(agent):
            return MemoryFootprint(session_count=0, total_bytes=0)
        return read_memory_footprint(resolve_codex_home(settings))

    @app.get("/api/agents/{agent_id}/health")
    async def agent_health_endpoint(agent_id: str) -> AgentHealth:
        """Read-only diagnostics probed for THIS agent's backend (a claude agent
        probes the claude CLI, not codex). Resolves the orchestrator too, so its
        settings page shares this endpoint. 404 unknown; never raises otherwise."""
        agent = _require_agent(agent_id)
        return await agent_health(settings, backend=agent.backend)

    @app.get("/api/agents/{agent_id}/account")
    def agent_account(agent_id: str) -> AccountInfo:
        """The account backing THIS agent: its effective model, auth mode, and
        (codex) usage quota. 404 unknown."""
        agent = _require_agent(agent_id)
        quota = (
            read_usage(resolve_codex_home(settings)) if _agent_is_codex(agent) else None
        )
        return AccountInfo(
            auth_mode=auth_mode_for_backend(settings, agent.backend),
            model=agent.model,
            enabled=settings.agent_enabled,
            quota=quota,
        )

    def _agent_detail_shell() -> Response:
        """Serve the agent-detail SPA shell; the client reads the id from the
        path. Registered before the static mount so `/agents/<id>` (and
        `/agents/<id>/settings`) route here while `/agents/` (the list) stays on
        the static index and `/api/...` is unaffected. 404 until the frontend is
        built. Not in the OpenAPI schema (it is a page, not an API)."""
        shell = settings.web_dist / "agent-detail.html"
        if not shell.is_file():
            raise HTTPException(status_code=404, detail="frontend not built")
        return FileResponse(shell, headers={"Cache-Control": "no-cache"})

    @app.get("/agents/{agent_id}", include_in_schema=False)
    def agent_detail_page(agent_id: str) -> Response:
        return _agent_detail_shell()

    @app.get("/agents/{agent_id}/{rest:path}", include_in_schema=False)
    def agent_detail_subpage(agent_id: str, rest: str) -> Response:
        return _agent_detail_shell()

    def _project_detail_shell() -> Response:
        """Serve the project-detail SPA shell; the client reads the id from the
        path. Registered before the static mount so `/projects/<id>` routes here
        while `/projects/` (the list) stays on the static index and `/api/...` is
        unaffected. 404 until the frontend is built. Not in the OpenAPI schema."""
        shell = settings.web_dist / "project-detail.html"
        if not shell.is_file():
            raise HTTPException(status_code=404, detail="frontend not built")
        return FileResponse(shell, headers={"Cache-Control": "no-cache"})

    @app.get("/projects/{project_id}", include_in_schema=False)
    def project_detail_page(project_id: str) -> Response:
        return _project_detail_shell()

    @app.get("/projects/{project_id}/{rest:path}", include_in_schema=False)
    def project_detail_subpage(project_id: str, rest: str) -> Response:
        return _project_detail_shell()

    @app.get("/api/agent/tools")
    async def get_agent_tools() -> list[AgentTool]:
        """The curated tools the agent can call (from the Scufris MCP server)."""
        from .mcp_server import mcp

        tools = await mcp.list_tools()
        disabled = set(settings.disabled_tools)
        result: list[AgentTool] = []
        for t in tools:
            schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
            props = schema.get("properties")
            args = list(props) if isinstance(props, dict) else []
            result.append(
                AgentTool(
                    name=t.name,
                    description=t.description or "",
                    server="scufris",
                    args=args,
                    parameters=_tool_parameters(t.inputSchema),
                    enabled=t.name not in disabled,
                )
            )
        return result

    @app.post("/api/agent/tools/{name}/run")
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

        from .mcp_server import mcp

        if name in set(settings.disabled_tools):
            raise HTTPException(status_code=403, detail=f"tool {name!r} is disabled")
        known = {t.name for t in await mcp.list_tools()}
        if name not in known:
            raise HTTPException(status_code=404, detail=f"unknown tool {name!r}")
        try:
            raw = await mcp.call_tool(name, req.args)
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

    @app.get("/api/agent/health")
    async def get_agent_health() -> AgentHealth:
        """Read-only diagnostics for the operator console (never raises)."""
        return await agent_health(settings)

    @app.get("/api/agent/sessions")
    def get_sessions() -> SessionsResponse:
        """List the agent's codex sessions (to switch between) + the current one."""
        if not settings.agent_enabled:
            return SessionsResponse(sessions=[], current=None)
        home = resolve_codex_home(settings)
        return SessionsResponse(
            sessions=list_sessions(home, os.getcwd()),
            current=agents.orchestrator_session_id(),
        )

    @app.post("/api/agent/session")
    async def post_session(action: SessionAction) -> CurrentSession:
        """Start a new session or switch to an existing one for the next turn."""
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        # Serialize on the orchestrator id (not the old "chat" key): its turns now
        # run through the supervisor keyed on that id, so a session switch cannot
        # interleave with an in-flight orchestrator turn.
        async with supervisor.serialized(ORCHESTRATOR_ID):
            if action.action == "switch":
                if not action.session_id:
                    raise HTTPException(
                        status_code=422, detail="session_id required to switch"
                    )
                agents.set_orchestrator_session(action.session_id)
            else:
                agents.set_orchestrator_session(None)
            return CurrentSession(current=agents.orchestrator_session_id())

    @app.post("/api/agent/session/fork")
    async def fork_session(request: ForkRequest) -> ForkResult:
        """Fork a conversation: start a new session seeded with the turns before
        the edited message plus the edited text, and run it as the first turn.

        codex-exec has no native branch, so the prior turns are pasted as context.
        """
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        home = resolve_codex_home(settings)
        messages = read_transcript(home, request.source_id)
        cut = max(0, request.message_index)
        seed = format_fork_seed(messages[:cut], request.text)
        # Drop the active session, then run the seed as a fresh turn. No outer
        # serialize() lock here: _launch_agent_turn already reserves the
        # orchestrator's serialize slot (and 409s a concurrent turn), so wrapping
        # this in supervisor.serialized(ORCHESTRATOR_ID) would self-deadlock on the
        # same key. The set-then-launch is synchronous, so nothing interleaves.
        agents.set_orchestrator_session(None)
        orchestrator = agents.get(ORCHESTRATOR_ID)
        _run_id, bus = _launch_agent_turn(orchestrator, None, seed)
        done = await _drain_turn(bus)
        return ForkResult(
            current=done.session_id or agents.orchestrator_session_id(),
            reply=done.reply,
        )

    @app.get("/api/agent/context")
    def get_context() -> SessionContext | None:
        """The current session's context snapshot (window + token usage + counts)."""
        if not settings.agent_enabled:
            return None
        return read_context(
            resolve_codex_home(settings), agents.orchestrator_session_id()
        )

    @app.get("/api/agent/session/{session_id}")
    def get_session_transcript(session_id: str) -> TranscriptResponse:
        """A session's past messages, so switching to it re-renders its history."""
        if not settings.agent_enabled:
            return TranscriptResponse(messages=[])
        home = resolve_codex_home(settings)
        return TranscriptResponse(messages=read_transcript(home, session_id))

    @app.delete("/api/agent/session/{session_id}")
    async def delete_agent_session(session_id: str) -> DeleteResult:
        """Delete a session (unlink its rollout); reset current if it was active."""
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with supervisor.serialized(ORCHESTRATOR_ID):
            deleted = delete_session(resolve_codex_home(settings), session_id)
            if deleted and agents.orchestrator_session_id() == session_id:
                agents.set_orchestrator_session(None)
            return DeleteResult(
                deleted=deleted, current=agents.orchestrator_session_id()
            )

    @app.get("/api/agent/usage")
    def get_usage() -> UsageQuota | None:
        """Account-wide usage/quota (the weekly rate-limit window)."""
        if not settings.agent_enabled:
            return None
        return read_usage(resolve_codex_home(settings))

    @app.get("/api/agent/memory")
    def get_memory() -> MemoryFootprint:
        """The agent's persistent footprint: codex rollout count/size/span."""
        if not settings.agent_enabled:
            return MemoryFootprint(session_count=0, total_bytes=0)
        return read_memory_footprint(resolve_codex_home(settings))

    @app.get("/api/agent/account")
    def get_account() -> AccountInfo:
        """The account backing the agent: auth mode, model, and usage quota."""
        quota = (
            read_usage(resolve_codex_home(settings)) if settings.agent_enabled else None
        )
        return AccountInfo(
            auth_mode=auth_mode_for_backend(settings, settings.agent_backend),
            model=settings.agent_model,
            enabled=settings.agent_enabled,
            quota=quota,
        )

    @app.post("/api/chat")
    async def post_chat(request: ChatRequest) -> AgentReply:
        """Send one message to the orchestrator and return its reply (turn-based).

        Runs through the SAME supervised backend path as any agent turn (B5bc):
        launch the orchestrator turn, then drain its event bus for the final
        reply. 503 when the agent is disabled, 409 when a turn is already active.
        """
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        orchestrator = agents.get(ORCHESTRATOR_ID)
        _run_id, bus = _launch_agent_turn(orchestrator, None, request.message)
        return (await _drain_turn(bus)).reply

    @app.post("/api/chat/stream")
    async def post_chat_stream(
        request: ChatRequest, http_request: Request
    ) -> StreamingResponse:
        """Send one message and stream live turn progress as SSE.

        The turn runs as a supervised BACKGROUND job (not inside this request):
        this endpoint starts it and relays its event bus. A dropped connection
        therefore does not cancel the turn, and there is no request timeout - the
        turn serializes on the "chat" key and is guarded by the run heartbeat.
        Each `data:` frame is a JSON stream event (`tool`, `text_delta`,
        `reasoning_delta`, `done`, `error`); each carries an SSE `id:` (the bus
        seq) so a reconnect can replay via `Last-Event-ID`.
        """
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")

        tmpdir: str | None = None
        image_paths: list[str] | None = None
        image_error: str | None = None
        if request.image is not None:
            try:
                tmpdir, path = _write_image_to_temp(request.image)
                image_paths = [path]
            except ValueError as exc:
                image_error = str(exc)

        # A bad image never launches a turn: relay a single error frame instead.
        if image_error is not None:

            async def error_events() -> AsyncIterator[str]:
                yield f":{' ' * 2048}\n\n"
                payload = json.dumps({"kind": "error", "detail": image_error})
                yield f"data: {payload}\n\n"

            return StreamingResponse(
                error_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        # The image tempdir is owned by the turn: cleaned when it finishes, NOT
        # when a relay disconnects.
        def cleanup() -> None:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

        orchestrator = agents.get(ORCHESTRATOR_ID)
        _run_id, bus = _launch_agent_turn(
            orchestrator,
            None,
            request.message,
            image_paths=image_paths,
            on_done=cleanup,
        )

        # Honour a reconnect: replay bus events newer than the client's last seq.
        after_seq = 0
        last_event_id = http_request.headers.get("last-event-id")
        if last_event_id and last_event_id.isdigit():
            after_seq = int(last_event_id)
        return _relay_bus_sse(bus, after_seq)

    @app.post("/api/chat/reset")
    async def post_chat_reset() -> dict[str, bool]:
        """Start a fresh conversation (forget prior context)."""
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with supervisor.serialized(ORCHESTRATOR_ID):
            agents.set_orchestrator_session(None)
        return {"ok": True}

    # Mount the built dashboard LAST so the /api routes above take precedence;
    # everything else falls through to the static bundle. Skipped (with a hint)
    # until the frontend has been built, so the API still runs standalone.
    if settings.web_dist.is_dir():
        app.mount(
            "/",
            _NoCacheStaticFiles(directory=settings.web_dist, html=True),
            name="web",
        )
    else:
        logger.warning(
            "web dist %s not found; serving API only. Build the frontend "
            "(cd web && npm install && npm run build) to serve the dashboard.",
            settings.web_dist,
        )

    # Group the API endpoints under OpenAPI tags so /docs (Swagger) and /redoc
    # render organized, labelled sections. Assigned by path (a single map in
    # `_route_tags`) instead of a `tags=` on every decorator.
    for route in app.routes:
        if isinstance(route, APIRoute) and not route.tags:
            route.tags = list(_route_tags(route.path))

    return app


def run_server(settings: Settings | None = None) -> None:
    """Launch the dashboard app with uvicorn."""
    import uvicorn

    settings = settings or Settings()
    # Un-forced: the CLI has usually already configured (honouring --debug); a
    # direct run_server() call configures from the setting instead.
    configure_logging(settings.log_level)
    logger.info(
        "starting scufris on %s:%d (agent %s)",
        settings.host,
        settings.port,
        "on" if settings.agent_enabled else "off",
    )
    # log_config=None: keep OUR logging config instead of uvicorn installing its
    # own, so scufris + uvicorn logs share one format/level.
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        log_level=settings.log_level.lower(),
    )
