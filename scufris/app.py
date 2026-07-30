"""The Scufris FastAPI application.

Serves a read-only JSON stats API and, when built, the static dashboard bundle.
The stats collector is injected so tests can supply a fake; production uses the
psutil-backed collector.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    TypeVar,
    cast,
)
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import sesh
from .agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamSessionStarted,
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
from .auth import (
    CSRF_COOKIE,
    CSRF_HEADER,
    PUBLIC_PATHS,
    PUBLIC_STATIC_PATHS,
    SESSION_COOKIE,
    UNSAFE_METHODS,
    LoginThrottle,
    SessionStore,
    auth_required,
    bearer_token,
    mint_api_token,
    operator_only,
    safe_next_path,
    same_origin,
    session_cookie_kwargs,
    token_matches,
    validate_auth_config,
    verify_password,
)
from .auth import (
    now as auth_now,
)
from .backends import get_backend, session_info
from .config import (
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
from .host import HostInspector, HostOverview
from .host_actions import (
    AlreadyDecided,
    Confirmation,
    HostActionRecord,
    HostActionStore,
    UnknownAction,
)
from .host_approvals import (
    CannotUndo,
    ConfirmationRequired,
    HostApprovalService,
    NoLiveRun,
    NotApplied,
    ProposalExpired,
    decision_message,
)
from .hostclient import (
    HostdClient,
    HostdError,
    HostdUnavailable,
    host_supervisor,
)
from .hostconfig import (
    ChangeState,
    ConfigBuildEvent,
    ConfigChange,
    ConfigChangeBuilder,
    ConfigChangeRefused,
    ConfigChangeStore,
    UnknownChange,
    config_supervisor,
    default_attr,
)
from .hostd.actions import ActionKind
from .hostd.audit import AuditRecord, Requester
from .hostd.protocol import ErrorCode
from .logsetup import configure_logging, new_request_id, set_request_id
from .mcp_common import api_token_var
from .mcp_models import AgentTool, McpServerHealth, ToolParam
from .metrics import Collector, HostStats, PsutilCollector
from .processes import ProcessCollector, ProcessList, PsutilProcessCollector
from .project_capabilities import (
    ProjectCapabilities,
    read_project_capabilities,
)
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
from .reasoning_store import ReasoningStore
from .sessions import (
    MemoryFootprint,
    SessionContext,
    SessionInfo,
    TranscriptMessage,
    UsageQuota,
    format_fork_seed,
    read_memory_footprint,
    read_usage,
    resolve_codex_home,
    strip_steering,
)
from .settings_store import (
    SettingsReadOnly,
    SettingsStore,
    UnknownSettingKey,
)
from .supervisor import AgentSupervisor, RunState, agent_supervisor
from .telegram import (
    ApprovalOps,
    ApprovalOutcome,
    OnCancel,
    OnMessageStream,
    OnReset,
    OrchestratorInfo,
    SettingsOps,
    TelegramBot,
)
from .version import scufris_version
from .wake import WakeBridge

logger = logging.getLogger(__name__)


SCUFRIS_VERSION = scufris_version()

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
    {
        "name": "auth",
        "description": "The operator session: log in, log out, and ask whether authentication is required at all.",
    },
    {
        "name": "host",
        "description": "Read-only host inspection: the live metrics snapshot (stats, processes) and the deeper overview - failed units, NixOS generations, storage and thermals.",
    },
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
        "description": "Agent configuration: effective config, the tool catalog and health checks.",
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
    if path.startswith("/api/auth/"):
        return ["auth"]
    if path in ("/api/stats", "/api/processes") or path.startswith("/api/host/"):
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
    # The dashboard polls the host overview on its own, much slower clock: that
    # endpoint shells out to systemctl and nixos-rebuild, and folding it into the
    # 2s stats poll would make the live gauges hostage to a subprocess.
    host_overview_seconds: float = 30.0


# Floor on the overview cache's TTL. The endpoint is subprocess-backed, so
# "uncached" is never a sensible configuration - a 0 in the env would otherwise
# turn every poll of every open tab into its own systemctl + nixos-rebuild
# fan-out, which is exactly what the cache exists to prevent.
MIN_HOST_OVERVIEW_TTL = 2.0


class _HostOverviewCache:
    """One slot holding the most recent host overview, with a TTL.

    The overview costs several subprocesses. Without this, every open dashboard
    tab (and every poll of every tab) would run its own `nixos-rebuild
    list-generations`. One slot, not a keyed dict: there is exactly one host, so
    there is nothing to key on and nothing to reap - the bounded-registry problem
    cannot arise here.
    """

    def __init__(
        self,
        inspector: HostInspector,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inspector = inspector
        self._ttl = max(MIN_HOST_OVERVIEW_TTL, ttl_seconds)
        # Injected so a test can assert the TTL BOUNDARY without sleeping on a
        # wall clock or monkeypatching the stdlib time module globally.
        self._clock = clock
        self._value: HostOverview | None = None
        self._collected_at = 0.0

    def fresh(self) -> HostOverview | None:
        """The cached value if it is still within the TTL, else None."""
        cached = self._value
        if cached is None:
            return None
        return cached if self._clock() - self._collected_at < self._ttl else None

    def get(self) -> HostOverview:
        """The cached overview, collecting it when the TTL has expired.

        NOT internally locked: single-flight is enforced by an ``asyncio.Lock``
        at the route, which suspends a waiting request instead of parking a
        thread on a mutex. A ``threading.Lock`` here would be held across the
        whole subprocess collection while every concurrent caller occupies one
        of the shared default-executor threads, so a slow `nixos-rebuild` would
        starve the executor the rest of the app uses.
        """
        cached = self.fresh()
        if cached is not None:
            return cached
        collected = self._inspector.overview()
        self._value = collected
        self._collected_at = self._clock()
        return collected


class _SseEvent(Protocol):
    """What the SSE relay needs of an event: that it can serialize itself.

    The relay is shared by agent turns and by host applies, which carry
    deliberately different event types (a root command's output must never be
    renderable as model text). This is the only thing the relay actually
    depends on.
    """

    def model_dump_json(self) -> str: ...


_SseEventT = TypeVar("_SseEventT", bound=_SseEvent)


def _last_event_id(request: Request) -> int:
    """The SSE seq a reconnecting client already has, 0 when it is new."""
    raw = request.headers.get("last-event-id")
    return int(raw) if raw and raw.isdigit() else 0


class LoginRequest(BaseModel):
    """The login body. Carries the password only - there is one operator, so
    there is no username to get wrong."""

    password: str


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
    the web and Telegram surfaces cannot each decide what "are you sure" means
    (tasks/20260729-125040/DECISION.md section 6).
    """

    acknowledge: str = ""


class ConfigChangeRequest(BaseModel):
    """Build a committed configuration and propose activating it.

    A REF, never a store path. Which revision gets built is a caller's to name -
    it is a commit in a repository, reviewable in git - but what that revision
    BUILDS INTO is resolved by this server, because a caller-supplied store path
    would mean the model choosing what gets activated
    (tasks/20260729-125035/DECISION.md section 2).
    """

    # Empty means the configured host config repo. A path may be a linked
    # worktree - which is where an agent will have been working.
    repo: str = ""
    # Empty means HEAD of that working tree.
    ref: str = ""
    # Which nixosConfiguration; empty means the configured one, then the hostname.
    attr: str = ""
    agent: str = ""
    run: str = ""


class HostActionLaunched(BaseModel):
    """What an approval returns: the record plus the run carrying it out."""

    action: HostActionRecord
    run_id: str


class AuthSession(BaseModel):
    """The authentication posture of the caller and of this deployment.

    `required` lets the frontend skip the login flow entirely in loopback
    development instead of guessing from a status code."""

    authenticated: bool
    required: bool


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


class PendingAgent(BaseModel):
    # An agent that needs the orchestrator (BC3): an unacknowledged WAITING
    # (request_input), REPORTED (report_back) or ERROR outcome. ``message`` is the
    # question / result summary / last message.
    agent_id: str
    state: AgentState
    message: str
    run_id: str
    session_id: str | None
    ts: float
    # Who/which orchestrator chat spawned this child (part 3); None = unattributed.
    parent_agent_id: str | None = None
    parent_session_id: str | None = None


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


def build_telegram_callbacks(
    settings: Settings,
    agents: AgentStore,
    supervisor: AgentSupervisor,
    launch_turn: Callable[..., tuple[str, EventBus[StreamEvent]]],
    active_run_id: Callable[[str], str | None],
) -> tuple[OnMessageStream, OnReset, OnCancel]:
    """Build the Telegram bot's orchestrator callbacks over the internal turn
    path (`_launch_agent_turn` + the run's EventBus), so the bot drives the SAME
    supervised orchestrator as the landing chat with no self-HTTP.

    ``on_message`` STREAMS the turn's ``StreamEvent`` values (the bot renders them
    message-per-phase). Every app-level condition - agent disabled, a 409 (a turn
    already active), a launch failure, or a backend ``StreamError`` - is mapped to
    a terminal ``StreamError`` whose ``detail`` is the friendly, user-facing line,
    so the raw technical detail never reaches the chat and a failed turn is always
    reported, never silently dropped.

    Module-level (not a `create_app` closure) so the stream/error behavior is
    unit-testable with a fake for `launch_turn`.
    """

    _FAILED = "Sorry - that turn failed. Please try again."

    async def on_message(text: str) -> AsyncIterator[StreamEvent]:
        """One orchestrator turn from a chat message -> a stream of StreamEvents."""
        if not settings.agent_enabled:
            yield StreamError(detail="The agent is disabled.")
            return
        orchestrator = agents.get(ORCHESTRATOR_ID)
        try:
            _run_id, bus = launch_turn(orchestrator, None, text)
        except HTTPException as exc:
            if exc.status_code == 409:
                yield StreamError(
                    detail="I'm still working on the previous message - "
                    "try again in a moment."
                )
                return
            logger.exception("telegram orchestrator turn failed (%s)", exc.status_code)
            yield StreamError(detail=_FAILED)
            return
        except Exception:
            logger.exception("telegram orchestrator turn errored")
            yield StreamError(detail=_FAILED)
            return
        try:
            async for _seq, event in bus.subscribe(after_seq=0):
                if isinstance(event, StreamError):
                    # Do not leak a raw backend detail to the chat; log it and
                    # surface the friendly line instead.
                    logger.warning("telegram orchestrator turn error: %s", event.detail)
                    yield StreamError(detail=_FAILED)
                    return
                yield event
                if isinstance(event, StreamDone):
                    return
        except Exception:
            logger.exception("telegram orchestrator stream errored")
            yield StreamError(detail=_FAILED)

    async def on_reset() -> None:
        """`/new`: forget the orchestrator's conversation, like /api/chat/reset.

        Serialized on ORCHESTRATOR_ID so a reset cannot interleave with an
        in-flight orchestrator turn (mirrors post_chat_reset)."""
        async with supervisor.serialized(ORCHESTRATOR_ID):
            agents.set_orchestrator_session(None)

    async def on_cancel() -> bool:
        """`/cancel`: stop the active orchestrator turn, like the web stop button."""
        run_id = active_run_id(ORCHESTRATOR_ID)
        return run_id is not None and supervisor.cancel(run_id)

    return on_message, on_reset, on_cancel


def create_app(
    collector: Collector | None = None,
    settings: Settings | None = None,
    process_collector: ProcessCollector | None = None,
    config_builder: ConfigChangeBuilder | None = None,
) -> FastAPI:
    """Build the app.

    ``config_builder`` is the seam for the NixOS build: tests inject one whose
    executor is scripted, because the real one spawns `nix build` and there is no
    honest way to fake a system build through a runner.
    """
    settings = settings or Settings()
    collector = collector or PsutilCollector()
    process_collector = process_collector or PsutilProcessCollector()
    projects = ProjectStore(settings)
    # First-class agents: named, project-bound records (A1). Running one is A3.
    # The landing orchestrator is a reserved record in this store (B5bc), so the
    # landing chat + session endpoints run through the same backend path as any
    # other agent - there is no longer a separate injected `Agent` object.
    agents = AgentStore(settings, projects)
    # Captures codex "thinking" from the live stream so a hard reload can re-show
    # the spoiler (reasoning is not recoverable from the rollout - see
    # reasoning_store). Written per turn in the turn stream, read at /transcript.
    reasoning_store = ReasoningStore(settings)

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
    supervisor = agent_supervisor(max_concurrent=settings.agent_max_concurrent)
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
        # The Telegram bot (if a token is configured) runs as a background task
        # for the app's lifetime, cancelled cleanly on shutdown. It is started
        # here rather than at create_app time so its poll loop lives on the
        # serving event loop. `_start_telegram_bot` is defined later in
        # create_app; the closure resolves it at call time.
        telegram_task = _start_telegram_bot()
        # Recover the approval queue from the helper before serving. The app's
        # registry is in-memory by design (the helper owns proposals), so without
        # this a restart inside a proposal's ten-minute window leaves a real pending
        # approval unreachable - the operator would see an empty queue while the
        # helper still held an appliable action (tasks/20260729-125040/DECISION.md
        # section 4). A helper that is not configured or not running is not an
        # error here: there is simply nothing to recover, and every host route
        # already answers "not configured" honestly.
        try:
            await approvals.refresh_pending()
        except (HostdUnavailable, HostdError) as exc:
            logger.info("could not recover the host approval queue: %s", exc)
        try:
            yield
        finally:
            if telegram_task is not None:
                telegram_task.cancel()
                with suppress(asyncio.CancelledError):
                    await telegram_task
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

    # --- authentication --------------------------------------------------
    #
    # Fail closed FIRST: a network-reachable bind with no credential must not
    # produce an app at all (see auth.validate_auth_config and
    # tasks/20260729-125015/DECISION.md). This raises AuthConfigError, which
    # `scufris serve` reports as a startup failure.
    validate_auth_config(settings)
    auth_on = auth_required(settings)
    app.state.auth_required = auth_on
    # The machine credential for THIS process's own tool subprocesses. It is put
    # on the app's own Settings, deliberately NOT in os.environ: an env var is
    # inherited by the agent CLI subprocess and hence by every shell command the
    # model runs, which would hand a sandboxed sub-agent the operator's full API
    # credential. From here it reaches exactly two places - the MCP servers that
    # call the API (agent.scufris_mcp_servers) and the in-process tool console
    # (which passes it through a ContextVar, since that tool's httpx call loops
    # back to this same server). Review round 1, finding 2.
    app.state.api_token = mint_api_token()
    settings.auth_api_token = app.state.api_token
    sessions = SessionStore(settings.state_dir / "auth_sessions.json")
    # Sweep once at startup so a restart clears out sessions that expired while
    # the server was down, rather than carrying them until each id is presented.
    sessions.prune(
        now=auth_now(),
        idle=settings.auth_session_idle_seconds,
        absolute=settings.auth_session_max_seconds,
    )
    app.state.sessions = sessions
    throttle = LoginThrottle(
        max_failures=settings.auth_login_max_failures,
        window_seconds=settings.auth_login_window_seconds,
    )

    def _deny(request: Request, status: int, detail: str) -> Response:
        """Refuse a request the way its caller can actually use.

        A browser NAVIGATION gets the login page (a bare 401 would show a blank
        screen); an API call gets a JSON status the frontend can react to. The
        redirect target is sanitized - it lands in a Location header, and an open
        redirect on a login page is a phishing primitive.
        """
        wants_html = "text/html" in request.headers.get("accept", "")
        if status == 401 and request.method == "GET" and wants_html:
            target = quote(safe_next_path(request.url.path), safe="/")
            return RedirectResponse(f"/login/?next={target}", status_code=303)
        return JSONResponse({"detail": detail}, status_code=status)

    @app.middleware("http")
    async def enforce_auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """The single enforcement point: deny by default, allow by exception.

        Registered BEFORE the request logger below so the logger stays outermost
        and a denial is still logged. Every route is gated unless its path is in
        the small public allowlist, so a route added tomorrow is protected by
        existing - `tests/test_auth.py` enumerates `app.routes` to prove it.

        Two identities are accepted. A browser presents the session cookie and is
        subject to the CSRF and origin checks (it carries ambient credentials that
        another site could try to ride). A machine caller - this app's own MCP
        tool subprocesses - presents the per-process bearer token and is not: it
        has no cookie to ride, and requiring a CSRF token would break every tool.
        """
        path = request.url.path
        # Operator-only paths are decided BEFORE the bearer branch AND before
        # the auth_on short-circuit, not inside either. Approving a privileged
        # host action is a human act; the machine token belongs to the app's own
        # tool subprocesses, which is to say to the agent. Deciding it later
        # would leave the framework's central claim untrue - and deciding it
        # only when auth is on would mean a loopback deployment lets an agent
        # approve its own proposal, which has nothing to do with the bind
        # address (see auth.OPERATOR_ONLY_PATTERN).
        if operator_only(path):
            # An operator-only path needs a real SESSION, and the check does not
            # look at the credential presented - it looks at whether one that
            # identifies a human was. The first version of this asked "is a
            # bearer token present?", which meant a caller that sent NO header at
            # all sailed through to the `auth_on` short-circuit below and
            # executed a root command anonymously. On loopback that is any
            # process on this machine, including the shell the model runs its own
            # commands in (`curl -XPOST .../approve`). Review round 1, R1.1.
            #
            # `validate_auth_config` refuses to build an app with host agency and
            # no operator credential, so on a correct deployment this branch is
            # about WHICH credential. It is written to stand alone anyway: a
            # guarantee that depends on a check somewhere else holding is not a
            # guarantee.
            session = sessions.get(
                request.cookies.get(SESSION_COOKIE),
                now=auth_now(),
                idle=settings.auth_session_idle_seconds,
                absolute=settings.auth_session_max_seconds,
            )
            if session is None:
                return _deny(
                    request,
                    403 if bearer_token(request.headers.get("authorization")) else 401,
                    "approving a host action needs an operator session; a machine "
                    "credential cannot do it and neither can an anonymous caller",
                )
            # Fully self-contained, including CSRF and origin - deliberately not
            # falling through to the generic block below, because that block is
            # skipped when auth is off and these paths must not be.
            if request.method in UNSAFE_METHODS:
                if not same_origin(
                    request.headers.get("origin"),
                    request.headers.get("referer"),
                    request.headers.get("host"),
                ):
                    return _deny(request, 403, "cross-origin request refused")
                if not token_matches(request.headers.get(CSRF_HEADER), session.csrf):
                    return _deny(request, 403, "missing or invalid CSRF token")
            return await call_next(request)
        if not auth_on:
            return await call_next(request)
        if path in PUBLIC_PATHS or path in PUBLIC_STATIC_PATHS:
            return await call_next(request)

        presented = bearer_token(request.headers.get("authorization"))
        if presented is not None:
            # No operator-only check here: the block above returned on every one
            # of those paths, whatever the credential and whatever the bind
            # address. There is ONE enforcement point, and a reader only has to
            # trust that one (review round 2, R2.6 removed the dead second).
            if token_matches(presented, app.state.api_token):
                return await call_next(request)
            return _deny(request, 401, "invalid credentials")

        session = sessions.get(
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        if session is None:
            return _deny(request, 401, "authentication required")
        if request.method in UNSAFE_METHODS:
            if not same_origin(
                request.headers.get("origin"),
                request.headers.get("referer"),
                request.headers.get("host"),
            ):
                return _deny(request, 403, "cross-origin request refused")
            if not token_matches(request.headers.get(CSRF_HEADER), session.csrf):
                return _deny(request, 403, "missing or invalid CSRF token")
        return await call_next(request)

    def _issue_session(response: Response, request: Request) -> None:
        """Mint a session and attach its cookies to ``response``.

        The session id ROTATES on every login (the caller revokes the old one
        first), which is what closes session fixation: an id an attacker planted
        in the browser before login is never the id that ends up authenticated.
        """
        session = sessions.create(now=auth_now())
        secure = request.url.scheme == "https"
        max_age = int(settings.auth_session_max_seconds)
        response.set_cookie(
            SESSION_COOKIE,
            session.id,
            **session_cookie_kwargs(secure=secure, max_age=max_age),
        )
        # Readable by JavaScript ON PURPOSE: the frontend echoes it back in the
        # CSRF header, and a cross-site attacker can send the cookie but cannot
        # read it to build the header.
        response.set_cookie(
            CSRF_COOKIE,
            session.csrf,
            **session_cookie_kwargs(secure=secure, max_age=max_age, http_only=False),
        )

    @app.post("/api/auth/login")
    async def post_auth_login(request: Request, body: LoginRequest) -> Response:
        """Exchange the operator password for a session.

        Public (it has to be), throttled per source, and deliberately uniform in
        its failure: a wrong password and an unconfigured credential answer the
        same way, so this endpoint cannot be used to probe the deployment.

        Origin-checked despite being public. Without it, any page the operator
        happens to visit can fire cross-origin logins at the dashboard's LAN
        address until the lockout window burns, denying the REAL operator their
        own login. The login page is same-origin, so nothing legitimate is
        affected - and the check runs BEFORE the throttle, so a refused
        cross-origin attempt cannot count toward the lockout it was trying to
        trigger. Review round 1, finding 5.
        """
        if not same_origin(
            request.headers.get("origin"),
            request.headers.get("referer"),
            request.headers.get("host"),
        ):
            return JSONResponse(
                {"detail": "cross-origin request refused"}, status_code=403
            )
        source = request.client.host if request.client else "unknown"
        moment = auth_now()
        if not throttle.allowed(source, now=moment):
            return JSONResponse(
                {"detail": "too many failed attempts; try again later"},
                status_code=429,
                headers={"Retry-After": str(throttle.retry_after(source, now=moment))},
            )
        stored = settings.auth_password_hash
        if not stored or not verify_password(body.password, stored):
            throttle.record_failure(source, now=moment)
            logger.warning("auth: failed login from %s", source)
            return JSONResponse({"detail": "invalid credentials"}, status_code=401)
        throttle.record_success(source)
        # Rotate: whatever session the browser was carrying is revoked, not reused.
        sessions.revoke(request.cookies.get(SESSION_COOKIE))
        # A login is the one moment a new record is added, so it is where the
        # store is swept for records nobody will ever present again.
        sessions.prune(
            now=moment,
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        response = JSONResponse({"authenticated": True})
        _issue_session(response, request)
        logger.info("auth: operator logged in from %s", source)
        return response

    @app.post("/api/auth/logout")
    async def post_auth_logout(request: Request) -> Response:
        """Revoke this session server-side and clear its cookies."""
        sessions.revoke(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        return response

    @app.get("/api/auth/session")
    async def get_auth_session(request: Request) -> AuthSession:
        """Whether this caller has a session, and whether one is needed at all.

        Public so the login page can ask without tripping a redirect loop. It
        reports posture only - never who, never the token."""
        if not auth_on:
            return AuthSession(authenticated=True, required=False)
        session = sessions.get(
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        return AuthSession(authenticated=session is not None, required=True)

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

    host_overview_cache = _HostOverviewCache(
        HostInspector(config_repo=settings.host_config_repo),
        settings.host_overview_seconds,
    )

    # Single-flight guard for the overview collection. An asyncio lock, not a
    # threading one: a waiting request suspends on the loop rather than holding
    # a default-executor thread, so a slow nixos-rebuild cannot starve the
    # executor that the rest of the app shares.
    host_overview_lock = asyncio.Lock()

    @app.get("/api/host/overview")
    async def get_host_overview() -> HostOverview:
        """The host's inspection snapshot: failed units, generations, storage,
        thermals.

        Cached for ``host_overview_seconds`` and collected off the event loop:
        it runs subprocesses (systemctl, nixos-rebuild), which must never block
        the loop serving the live stats poll.
        """
        cached = host_overview_cache.fresh()
        if cached is not None:
            return cached
        async with host_overview_lock:
            # Re-check inside the lock: a burst that queued here while the first
            # request collected must serve that result, not collect again.
            return await asyncio.to_thread(host_overview_cache.get)

    # --- privileged host actions -----------------------------------------
    #
    # propose -> preview -> approve -> apply -> audit -> roll back. The verbs,
    # the previews, the proposals and the audit log all live in the root helper
    # (scufris.hostd); this is the operator-facing surface over its socket.
    #
    # The approval endpoints are in auth.OPERATOR_ONLY_PATTERN, so the machine
    # bearer token - which the app's own agent subprocesses hold - is refused
    # there before the middleware's short-circuit. An agent may propose. Only a
    # human with a session may approve.

    hostd = HostdClient(settings.hostd_socket, settings.hostd_secret)
    host_actions = HostActionStore()
    # One at a time: two root commands running concurrently on one machine is
    # not something an operator approved.
    host_supervisor_ = host_supervisor(max_concurrent=1)
    # The ONE decision path. These routes are one surface over it; the Telegram bot
    # is the other, and it calls the same methods with a chat-derived actor. Every
    # rule after "who is deciding" lives in the service, so the two cannot drift
    # (tasks/20260729-125040/DECISION.md section 3).
    approvals = HostApprovalService(
        hostd=hostd, actions=host_actions, supervisor=host_supervisor_
    )
    app.state.hostd = hostd
    app.state.host_actions = host_actions
    app.state.host_supervisor = host_supervisor_
    app.state.host_approvals = approvals

    def _caller_is_agent(request: Request) -> bool:
        """Whether this caller is one of the app's own tool subprocesses (an AGENT)
        rather than the operator.

        Derived from the CREDENTIAL, never from the body - the same rule
        ``_requester_identity`` follows and for the same reason: "who is asking" is
        exactly the question a caller must not be able to answer about itself. A
        session is the operator; a bearer token is a machine, which is to say an
        agent; neither (only reachable with auth off) is nobody, and nobody is not
        an agent.
        """
        session = sessions.get(
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        if session is not None:
            return False
        return bearer_token(request.headers.get("authorization")) is not None

    def _operator_identity(request: Request) -> str:
        """Who approved, for the record. One operator, so this is traceability."""
        session = sessions.get(
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        return f"operator:{session.id[:8]}" if session is not None else "operator"

    def _requester_identity(
        request: Request, *, agent: str = "", run: str = ""
    ) -> Requester:
        """Who asked, derived from the CREDENTIAL rather than from the body.

        "Who asked" is the one question the audit exists to answer, so it must
        not be answerable by the caller. The first version read
        `actor = "agent" if body.agent else ...`, and the MCP tool sent no
        `agent` field - so every agent-originated proposal was written into the
        root-owned log as having been asked for by the operator (review round 1,
        R1.6). A body field is a hint about WHICH agent; the credential is the
        fact about what kind of caller it is.
        """
        session = sessions.get(
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=settings.auth_session_idle_seconds,
            absolute=settings.auth_session_max_seconds,
        )
        if session is not None:
            return Requester(actor=f"operator:{session.id[:8]}", agent=agent, run=run)
        if bearer_token(request.headers.get("authorization")) is not None:
            # A machine credential: this app's own tool subprocess, which is to
            # say an agent. It may name itself, but it cannot claim to be human.
            return Requester(actor="agent", agent=agent or "orchestrator", run=run)
        # Neither: only reachable with auth off, where the caller is anonymous
        # and the record should say exactly that rather than guess.
        return Requester(actor="unauthenticated", agent=agent, run=run)

    def _host_action_or_404(action_id: str) -> HostActionRecord:
        try:
            return host_actions.get(action_id)
        except UnknownAction:
            raise HTTPException(status_code=404, detail="no such host action") from None

    def _hostd_http_error(exc: Exception) -> HTTPException:
        """Map the helper's own refusals onto statuses a client can act on."""
        if isinstance(exc, HostdUnavailable):
            return HTTPException(status_code=503, detail=str(exc))
        if isinstance(exc, HostdError):
            status = {
                ErrorCode.NOT_FOUND: 404,
                ErrorCode.EXPIRED: 409,
                ErrorCode.DRIFTED: 409,
                ErrorCode.ALREADY_USED: 409,
                ErrorCode.REFUSED: 422,
                ErrorCode.BAD_REQUEST: 422,
                ErrorCode.UNAUTHORIZED: 502,
            }.get(exc.code, 502)
            return HTTPException(status_code=status, detail=exc.detail)
        return HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/host/actions", status_code=201)
    async def propose_host_action(
        body: HostActionRequest, request: Request
    ) -> HostActionRecord:
        """Ask the helper to preview an action. Proposing changes nothing.

        Open to an agent as well as the operator: proposing is how an assistant
        asks. It is the APPROVAL that is a human act, and that is a different
        endpoint with a different credential requirement.

        ``activate`` is refused here, and that refusal is load-bearing rather than
        tidiness: its argument is a store path, so accepting one would let the
        caller choose which system this machine boots and reduce the closure diff
        to a faithful description of the caller's own choice. The only route to an
        activation is /api/host/config/changes, which builds the path itself from
        a revision it resolved (tasks/20260729-125035/DECISION.md section 2).
        """
        if body.kind is ActionKind.ACTIVATE:
            raise HTTPException(
                status_code=422,
                detail=(
                    "activate is not proposed directly: it names a store path, and "
                    "what gets activated must be something this server built from "
                    "an identified commit. Post the ref to "
                    "/api/host/config/changes instead - it builds, diffs and then "
                    "proposes the activation for you."
                ),
            )
        try:
            proposal = await hostd.propose(
                body.kind,
                body.args,
                _requester_identity(request, agent=body.agent, run=body.run),
            )
        except (HostdUnavailable, HostdError) as exc:
            raise _hostd_http_error(exc) from exc
        return approvals.record_proposal(proposal)

    @app.get("/api/host/actions")
    async def list_host_actions() -> list[HostActionRecord]:
        """The proposal queue, newest first.

        Reconciles with the helper first (throttled), so the queue shows proposals
        this process did not create - one made before a restart, or by another client
        of the same socket - rather than only what it happens to remember.
        """
        try:
            await approvals.refresh_pending(min_interval=settings.host_queue_refresh_seconds)
        except (HostdUnavailable, HostdError) as exc:
            logger.debug("queue reconcile skipped: %s", exc)
        return host_actions.list()

    @app.get("/api/host/audit")
    async def get_host_audit(limit: int = 50) -> list[AuditRecord]:
        """The helper's own audit tail: what was requested, refused and applied."""
        try:
            return await hostd.audit_tail(max(1, min(500, limit)))
        except (HostdUnavailable, HostdError) as exc:
            raise _hostd_http_error(exc) from exc

    @app.get("/api/host/actions/{action_id}")
    async def get_host_action(action_id: str) -> HostActionRecord:
        return _host_action_or_404(action_id)

    @app.get("/api/host/actions/{action_id}/events")
    async def host_action_events(
        action_id: str, http_request: Request
    ) -> StreamingResponse:
        """Relay an approved action's live output as SSE."""
        record = _host_action_or_404(action_id)
        bus = host_supervisor_.bus(record.run_id) if record.run_id else None
        if bus is None:
            raise HTTPException(status_code=404, detail="this action has no live run")
        return _relay_bus_sse(bus, _last_event_id(http_request))

    @app.post("/api/host/actions/{action_id}/approve")
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
            record, run_id = await approvals.approve(
                action_id,
                actor=_operator_identity(request),
                acknowledge=decision.acknowledge,
            )
        except UnknownAction:
            raise HTTPException(
                status_code=404, detail="no such host action"
            ) from None
        except ConfirmationRequired as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (AlreadyDecided, ProposalExpired) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise _hostd_http_error(exc) from exc
        return HostActionLaunched(action=record, run_id=run_id)

    @app.get("/api/host/actions/{action_id}/confirmation")
    async def get_host_action_confirmation(action_id: str) -> Confirmation:
        """What approving this action requires: its risk class in words, the undo
        sentence (or the statement that there is none), and - for a one-way action -
        the acknowledgement token an approve must carry.

        Both approval surfaces render THIS rather than deciding for themselves how
        much friction an action deserves; it is also carried inline on every record
        in the queue listing, so a client needs this route only for a single action.
        """
        try:
            return approvals.confirmation(action_id)
        except UnknownAction:
            raise HTTPException(
                status_code=404, detail="no such host action"
            ) from None

    @app.post("/api/host/actions/{action_id}/cancel")
    async def cancel_host_action(action_id: str, request: Request) -> HostActionRecord:
        """Stop an apply that is running. OPERATOR ONLY.

        Cancelling reaches the helper, which signals the whole process group and
        records the cancellation - so the outcome is a recorded fact rather than
        an unknown state. Whatever the command had already done still stands,
        and the record says so.
        """
        try:
            return approvals.cancel(action_id)
        except UnknownAction:
            raise HTTPException(
                status_code=404, detail="no such host action"
            ) from None
        except NoLiveRun as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/host/actions/{action_id}/deny")
    async def deny_host_action(
        action_id: str, body: HostDecisionRequest, request: Request
    ) -> HostActionRecord:
        """Refuse a proposal, burning it. OPERATOR ONLY.

        The reason is not decoration: it is what reaches the agent that asked, so
        it can adapt instead of proposing the same thing again.
        """
        try:
            return await approvals.deny(
                action_id,
                actor=_operator_identity(request),
                reason=body.reason,
            )
        except UnknownAction:
            raise HTTPException(
                status_code=404, detail="no such host action"
            ) from None
        except AlreadyDecided as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise _hostd_http_error(exc) from exc

    @app.post("/api/host/actions/{action_id}/revert", status_code=201)
    async def revert_host_action(action_id: str, request: Request) -> HostActionRecord:
        """Propose the inverse of an applied action. OPERATOR ONLY.

        An undo is itself a host action: it gets its own preview and its own
        approval. Nothing is reverted by this call - the reversal is proposed,
        and the operator approves it like any other change.
        """
        try:
            return await approvals.revert(
                action_id, actor=_operator_identity(request)
            )
        except UnknownAction:
            raise HTTPException(
                status_code=404, detail="no such host action"
            ) from None
        except CannotUndo as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NotApplied as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostdUnavailable, HostdError) as exc:
            raise _hostd_http_error(exc) from exc

    # --- NixOS configuration changes (R3) --------------------------------
    #
    # The configuration repository is a PROJECT: an agent edits and commits it
    # through the ordinary project machinery, and none of that happens here. What
    # happens here is the last mile - resolve a ref to a commit, build it as the
    # operator, and hand the resulting store path to the helper as an `activate`
    # proposal. See tasks/20260729-125035/DECISION.md.

    config_changes = ConfigChangeStore()
    config_builder = config_builder or ConfigChangeBuilder(
        build_timeout=settings.host_config_build_timeout
    )
    # Its own supervisor: a NixOS build can run for an hour and needs no
    # privilege, so it must not sit in the single slot that serializes approved
    # root commands.
    config_supervisor_ = config_supervisor(max_concurrent=1)
    app.state.config_changes = config_changes
    app.state.config_supervisor = config_supervisor_

    def _config_change_or_404(change_id: str) -> ConfigChange:
        try:
            return config_changes.get(change_id)
        except UnknownChange:
            raise HTTPException(
                status_code=404, detail="no such config change"
            ) from None

    @app.post("/api/host/config/changes", status_code=201)
    async def propose_config_change(
        body: ConfigChangeRequest, request: Request
    ) -> ConfigChange:
        """Build a committed configuration and propose activating it.

        Open to an agent, like every other propose path: building changes nothing
        and the activation it produces still needs the operator. The build runs as
        this process's user - never root - and reads the tree from the COMMIT, so
        this endpoint cannot touch the repository's working tree.
        """
        repo = Path(body.repo).expanduser() if body.repo else settings.host_config_repo
        attr = body.attr or settings.host_config_attr or default_attr()
        requester = _requester_identity(request, agent=body.agent, run=body.run)
        try:
            # git reads only: milliseconds, so the request answers immediately.
            # The flake evaluation and the build both happen in the run.
            _main, resolved = await asyncio.to_thread(
                config_builder.resolve,
                repo,
                body.ref or "HEAD",
                allowed=settings.host_config_repo,
            )
        except ConfigChangeRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        in_flight = config_changes.building_for(resolved.repo)
        if in_flight is not None:
            # Refused, not queued: two builds of one repository contend for the
            # same evaluation and store, and a queued NixOS build sits for an
            # hour with no visible reason.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"a configuration build is already running for {resolved.repo} "
                    f"({in_flight.id}, {in_flight.resolved.ref}). Wait for it or "
                    "cancel it before starting another."
                ),
            )
        change = config_changes.put(
            ConfigChange(
                id=uuid.uuid4().hex,
                resolved=resolved,
                attr=attr,
                agent=requester.agent,
                requested_by=requester.actor,
            )
        )
        change.run_id = f"config:{change.id}"

        async def _propose(built: ConfigChange) -> str:
            proposal = await hostd.propose(
                ActionKind.ACTIVATE,
                {
                    "toplevel": built.toplevel,
                    "repo": built.resolved.repo,
                    "rev": built.resolved.rev,
                },
                requester,
            )
            # Through the service, like every other proposal: a configuration
            # activation waiting on the operator must mark the agent that asked for
            # it as BLOCKED and reach the operator's surfaces the same way a unit
            # restart does.
            approvals.record_proposal(proposal)
            return proposal.id

        def _stream() -> AsyncIterator[ConfigBuildEvent]:
            return config_builder.stream(change, _propose)

        config_supervisor_.start(
            change.run_id,
            _stream,
            # One build at a time per repository. The refusal above is the
            # visible half; this is what makes it true even in a race.
            serialize_key=f"config:{resolved.repo}",
        )
        return change

    @app.get("/api/host/config/changes")
    async def list_config_changes() -> list[ConfigChange]:
        """Configuration changes, newest first."""
        return config_changes.list()

    @app.get("/api/host/config/changes/{change_id}")
    async def get_config_change(change_id: str) -> ConfigChange:
        return _config_change_or_404(change_id)

    @app.get("/api/host/config/changes/{change_id}/events")
    async def config_change_events(
        change_id: str, http_request: Request
    ) -> StreamingResponse:
        """Relay a build's live log as SSE."""
        change = _config_change_or_404(change_id)
        bus = config_supervisor_.bus(change.run_id) if change.run_id else None
        if bus is None:
            raise HTTPException(status_code=404, detail="this change has no live build")
        return _relay_bus_sse(bus, _last_event_id(http_request))

    @app.post("/api/host/config/changes/{change_id}/cancel")
    async def cancel_config_change(change_id: str, request: Request) -> ConfigChange:
        """Stop a build that is running.

        Not operator-only: a build holds no privilege and stopping it undoes
        nothing. What IS operator-only is approving the activation it produces.
        """
        change = _config_change_or_404(change_id)
        if change.state is not ChangeState.BUILDING or not change.run_id:
            raise HTTPException(
                status_code=409, detail="this change has no running build to cancel"
            )
        if not config_supervisor_.cancel(change.run_id):
            raise HTTPException(
                status_code=409, detail="this change has no running build to cancel"
            )
        return change

    @app.get("/api/config")
    def get_config() -> AppConfig:
        """Client-facing knobs: poll intervals and whether the agent is on."""
        return AppConfig(
            poll_seconds=settings.poll_seconds,
            agent_enabled=settings.agent_enabled,
            # The floored value, so the client polls at the cadence the server
            # actually refreshes at rather than one the cache will not honour.
            host_overview_seconds=max(
                MIN_HOST_OVERVIEW_TTL, settings.host_overview_seconds
            ),
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
        return AgentConfig(
            enabled=settings.agent_enabled,
            backend=settings.agent_backend,
            model=settings.agent_model,
            auth_mode=auth_mode_for_backend(settings, settings.agent_backend),
            tools_enabled=settings.agent_tools_enabled,
            sandbox="read-only",
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
        updates = update.model_dump(exclude_none=True)
        try:
            store.apply(updates)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (UnknownSettingKey, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _agent_config()

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

    @app.get("/api/agents/pending")
    def list_pending_agents(
        parent_session_id: str | None = None,
    ) -> list[PendingAgent]:
        """The agents that need the orchestrator (BC3): those with an
        unacknowledged needs-input (WAITING, from request_input), reported-done
        (REPORTED, from report_back) or ERROR outcome, newest first. The
        orchestrator polls this to find blocked or finished sub-agents. Declared
        before /api/agents/{id} (like /api/agents/backends) so "pending" is not
        parsed as an agent id.

        ``parent_session_id`` scopes to one orchestrator chat (part 3): the result
        keeps children that chat spawned PLUS unattributed ones (UI-launched, or
        spawned before a fresh turn had a session), and drops children clearly
        owned by a DIFFERENT chat - so a poll from chat A never sees chat B's
        children, and nothing is orphaned. Each row is annotated with its parent."""
        pending = agents.pending_outcomes()
        rows: list[PendingAgent] = []
        for agent_id, o in pending.items():
            parent_agent, parent_sess = agents.parent_of(agent_id)
            # Scope: keep this chat's own children and unattributed ones; drop
            # another chat's. No filter (None query) -> keep all (back-compat).
            if (
                parent_session_id is not None
                and parent_sess is not None
                and parent_sess != parent_session_id
            ):
                continue
            rows.append(
                PendingAgent(
                    agent_id=agent_id,
                    state=o.state,
                    message=o.message,
                    run_id=o.run_id,
                    session_id=o.session_id,
                    ts=o.ts,
                    parent_agent_id=parent_agent,
                    parent_session_id=parent_sess,
                )
            )
        rows.sort(key=lambda r: r.ts, reverse=True)
        return rows

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
    ) -> tuple[str, EventBus[StreamEvent]]:
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
        # Accumulate the turn's live reasoning ("thinking") deltas so the sidecar
        # can persist them for reload survival (codex-only; reasoning is absent
        # from the rollout). Codex is the only backend that streams these.
        is_codex = canonical_backend(agent.backend) == "codex"
        reasoning_parts: list[str] = []

        async def turn_stream() -> AsyncIterator[StreamEvent]:
            async for event in backend.stream(
                settings,
                prompt,
                session_id=agent.session_id,
                cwd=project.cwd if project is not None else None,
                image_paths=image_paths,
                permission_mode=agent.permission_mode,
                is_orchestrator=agent.id == ORCHESTRATOR_ID,
                agent_id=agent.id,
            ):
                if isinstance(event, StreamReasoningDelta):
                    reasoning_parts.append(event.delta)
                if isinstance(event, StreamSessionStarted):
                    # Record ownership the moment the session id is known (turn-start,
                    # before the turn streams), so a client refreshing mid-turn sees
                    # the session instead of nothing until mark_finished. Keyed under
                    # the launch-time backend snapshot; idempotent with the terminal
                    # mark_finished. Also seeds captured[] so an errored turn (no
                    # done frame) still persists the id.
                    captured["session_id"] = event.session_id
                    agents.record_running_session(
                        agent.id, agent.backend, event.session_id
                    )
                if isinstance(event, StreamDone):
                    if event.session_id:
                        captured["session_id"] = event.session_id
                    # The final reply text is the durable outcome's message (BC1),
                    # so the orchestrator can read what a finished agent said/asked
                    # after the per-run bus has closed.
                    captured["message"] = event.reply.text
                    # Persist this turn's reasoning to the sidecar BEFORE yielding
                    # the done frame, so a reload the client triggers on `done`
                    # reads a sidecar that is already written (the on_complete
                    # persist callback runs in the supervisor's finally, AFTER the
                    # frame - too late; lesson out-of-context-review-misses-cross-
                    # layer-timing). One entry per completed codex turn, keyed by
                    # the just-captured session id; keeps the sidecar 1:1 with the
                    # assistant messages read_transcript surfaces (empty reasoning
                    # when the model did not think). Non-codex turns never stream
                    # reasoning, so nothing is written.
                    if is_codex and event.reply.text.strip():
                        reasoning_store.append(
                            captured.get("session_id"),
                            "".join(reasoning_parts),
                            answer=event.reply.text,
                        )
                yield event

        run_id = f"{agent.id}:{uuid.uuid4().hex}"

        def persist(run_state: RunState) -> None:
            # Best-effort: if the agent was deleted mid-run, mark_finished raises
            # AgentNotFound, which the supervisor swallows and logs - the terminal
            # state just is not persisted for a record that no longer exists.
            #
            # A turn FAILED when the supervisor's RunPhase is not DONE (the
            # except-clause paths: cancelled / stall / budget / crash) OR when a
            # backend yielded a terminal StreamError (RunPhase stays DONE but
            # _drain recorded the detail on run.error). Either way the agent's
            # terminal state is ERROR, and the diagnostic detail is the durable
            # outcome message so agent_status / pending_agents can report WHY. The
            # error detail WINS over any captured reply on a failed turn: a rogue
            # backend that emits both a done frame and a trailing StreamError must
            # still surface the failure, not a stale success reply.
            # A user-initiated cancel is its OWN terminal state, distinct from a
            # crash/stall/backend-error: it must read as a neutral stop (not a red
            # ERROR) and must NOT surface in pending_agents as an agent needing the
            # orchestrator. Keyed off the explicit run.cancelled flag, not the
            # "cancelled" error string, so a real error is never misclassified.
            failed = run_state.state != RunPhase.DONE or bool(run_state.error)
            if run_state.cancelled:
                terminal_state = AgentState.CANCELLED
                message = "cancelled"
            elif failed:
                terminal_state = AgentState.ERROR
                message = run_state.error or captured.get("message", "")
            else:
                terminal_state = AgentState.DONE
                message = captured.get("message", "")
            agents.mark_finished(
                agent.id,
                state=terminal_state,
                session_id=captured.get("session_id"),
                # Key the session under the backend this turn RAN on (the
                # launch-time snapshot), not whatever the current config is - a
                # backend switch that raced the turn must not mislabel it.
                backend=agent.backend,
                message=message,
                run_id=run_id,
            )
            # Turn-owned cleanup (e.g. an attached image tempdir) runs when the
            # run ends, not when a relay disconnects.
            if on_done is not None:
                on_done()
            # Wake bridge (BC4): a sub-agent that finished needing input wakes the
            # orchestrator; the orchestrator's OWN completion drains any deferred
            # wakes. Runs AFTER mark_finished so the outcome it reads is current;
            # fires here (the finally, past the run's serialize-key release) so a
            # launch never holds ORCHESTRATOR_ID. auto_wake off -> no-op.
            wake_bridge.on_run_complete(agent.id)
            # A host-action decision that arrived while this agent was mid-turn is
            # delivered now, for the same reason the wake fires here: the finishing
            # run's serialize key is released, so launching a turn for this agent
            # cannot deadlock on it (`serialize-then-launch-self-deadlocks-on-shared-key`).
            _drain_deferred_decision(agent.id)

        agent_runs[agent.id] = run_id
        agents.mark_running(agent.id)
        bus = supervisor.start(
            run_id,
            turn_stream,
            serialize_key=agent.id,
            budget_seconds=None,
            heartbeat_seconds=settings.agent_heartbeat_seconds,
            on_complete=persist,
            # Expose the raw turn prompt on the run's status so a client
            # reattaching mid-turn can render the user bubble before the backend
            # flushes it to its durable log (the steering added downstream is
            # stripped at the status read boundary).
            prompt=prompt,
        )
        return run_id, bus

    # --- a pending approval is a BLOCKED agent -----------------------------
    #
    # The requesting agent proposes and ends its turn; the operator decides; the
    # decision resumes the agent. That round trip runs on the machinery a sub-agent
    # already uses (the outcome store plus one launched turn), with ONE difference
    # that matters: the state is BLOCKED, not WAITING, because the decider is the
    # operator and not the orchestrator (tasks/20260729-125040/DECISION.md
    # section 5).

    # agent_id -> the decision text that could not be delivered because a turn was
    # in flight. Drained by the run-completion callback, like a deferred wake.
    deferred_decisions: dict[str, str] = {}

    def _requesting_agent(record: HostActionRecord) -> AgentRecord | None:
        """The agent whose proposal this is, if an agent asked at all.

        The operator proposing from a surface has no agent to block, and the
        ORCHESTRATOR is never it: it holds no propose tool, and the identity helper
        labels a nameless machine caller "orchestrator" by default, so a proposal
        attributed to it means "some agent-credentialled caller that did not name
        itself" rather than a resumable agent turn.
        """
        agent_id = record.proposal.requester.agent.strip()
        if not agent_id or agent_id == ORCHESTRATOR_ID:
            return None
        try:
            return agents.get(agent_id)
        except AgentNotFound:
            return None

    def _mark_requester_blocked(record: HostActionRecord) -> None:
        """Record the requesting agent as BLOCKED on this proposal."""
        agent = _requesting_agent(record)
        if agent is None:
            return
        agents.awaiting_approval(
            agent.id,
            f"waiting for the operator to decide host action {record.id}: "
            f"{record.proposal.summary}",
            run_id=agent_runs.get(agent.id, ""),
            session_id=agent.session_id,
        )

    def _deliver_decision(agent: AgentRecord, text: str) -> None:
        """Resume the agent with the decision, or hold it until its turn ends.

        A 409 means a turn for that agent is already in flight (it proposed and kept
        working), and dropping the decision there would be the exact failure the
        denial path exists to prevent - so it is held and delivered by the
        completion callback instead.
        """
        try:
            project = projects.get(agent.project_id) if agent.project_id else None
        except ProjectNotFound:
            project = None
        try:
            _launch_agent_turn(agent, project, text)
        except HTTPException:
            held = deferred_decisions.get(agent.id)
            deferred_decisions[agent.id] = f"{held}\n\n{text}" if held else text

    def _tell_requester_the_decision(record: HostActionRecord) -> None:
        """Hand a decided action's outcome back to the agent that asked for it."""
        agent = _requesting_agent(record)
        if agent is None:
            return
        text = decision_message(record)
        if text is None:
            return  # approved and still running: the result is the news
        _deliver_decision(agent, text)

    def _drain_deferred_decision(agent_id: str) -> None:
        """Deliver a decision that was held while the agent was mid-turn."""
        text = deferred_decisions.pop(agent_id, None)
        if text is None:
            return
        try:
            agent = agents.get(agent_id)
        except AgentNotFound:
            return
        _deliver_decision(agent, text)

    def _telegram_announce(
        record: HostActionRecord, *, decision: bool
    ) -> None:
        """Push a proposal, or a decision, into the operator's chat.

        Fire-and-forget on purpose: this is a NOTIFICATION, and a Telegram outage
        must not fail the decision that already happened or the proposal that is
        already in the queue. The hook layer logs whatever this raises.

        A restored proposal (recovered from the helper after a restart) deliberately
        does NOT come through here - see the on_restored wiring below: re-announcing
        old news on every restart is how a notification channel gets muted.
        """
        bot = getattr(app.state, "telegram_bot", None)
        if bot is None:
            return
        coroutine = (
            bot.announce_decision(record) if decision else bot.announce_proposal(record)
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: a store driven directly by a test or a CLI, where there is
            # no bot to notify anyway.
            coroutine.close()
            return
        task = loop.create_task(coroutine)
        # Held until it finishes, so the task is not garbage-collected mid-send.
        _notify_tasks.add(task)
        task.add_done_callback(_notify_tasks.discard)

    _notify_tasks: set[asyncio.Task[None]] = set()

    approvals.on_proposed(_mark_requester_blocked)
    approvals.on_proposed(lambda record: _telegram_announce(record, decision=False))
    approvals.on_decided(lambda record: _telegram_announce(record, decision=True))

    # A proposal recovered from the helper after a restart marks its requester too:
    # that agent IS still waiting, and its persisted outcome should say so rather
    # than depending on which process wrote it.
    approvals.on_restored(_mark_requester_blocked)
    approvals.on_decided(_tell_requester_the_decision)

    def _orchestrator_busy() -> bool:
        """Whether the orchestrator has a queued/running turn (the same condition
        `_launch_agent_turn` 409s on)."""
        run_id = agent_runs.get(ORCHESTRATOR_ID)
        state = supervisor.status(run_id) if run_id else None
        return state is not None and state.state in (RunPhase.QUEUED, RunPhase.RUNNING)

    def _wake_launch(prompt: str) -> bool:
        """Grant the orchestrator one turn carrying ``prompt`` (resuming its
        session); True if granted, False if it turned out busy (409 race). Called
        from the completion callback, which has already released the finishing
        run's serialize key - so this never holds ORCHESTRATOR_ID (no self-deadlock,
        lesson `serialize-then-launch-self-deadlocks-on-shared-key`)."""
        try:
            _launch_agent_turn(agents.get(ORCHESTRATOR_ID), None, prompt)
            return True
        except HTTPException:
            return False

    wake_bridge = WakeBridge(
        agents=agents,
        settings=settings,
        is_orchestrator_busy=_orchestrator_busy,
        launch=_wake_launch,
    )

    async def _drain_turn(bus: EventBus[StreamEvent]) -> StreamDone:
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

    def _build_telegram_approval_ops() -> ApprovalOps:
        """The bot's host-approval providers, wired to the ONE approval service.

        Two things live here rather than in the transport, and both are the same
        rule the web routes follow:

        - the ACTOR is derived, never supplied. The bot hands over a chat id; this
          builds `operator:telegram:<chat_id>`, so the audit says which surface
          decided and a transport cannot claim to be someone else (the web path
          derives its actor from the session cookie for the same reason).
        - the allowlist is re-checked HERE. The bot already refuses a chat that is
          not allowlisted, and this refuses it again, so neither layer is the only
          thing between a stray chat and a root command.

        Every refusal message the operator reads is the service's own sentence
        ("already denied by ...", "this proposal has expired", "needs the explicit
        acknowledgement ..."), which is what keeps the two surfaces from developing
        different ideas of the same rule.
        """

        def _actor(chat_id: int) -> str:
            return f"operator:telegram:{chat_id}"

        def _refuse_unallowed(chat_id: int) -> ApprovalOutcome | None:
            if chat_id in set(settings.telegram_allowed_chat_ids):
                return None
            logger.warning(
                "refused a host decision from a non-allowlisted telegram chat %s",
                chat_id,
            )
            return ApprovalOutcome(
                ok=False, message="this chat cannot decide host actions"
            )

        async def pending() -> list[HostActionRecord]:
            # Reconcile with the helper first, so a proposal made before a restart
            # (or by another client of the socket) is decidable from the phone too.
            try:
                await approvals.refresh_pending(
                    min_interval=settings.host_queue_refresh_seconds
                )
            except (HostdUnavailable, HostdError) as exc:
                logger.debug("telegram queue reconcile skipped: %s", exc)
            # `decidable`, not "pending": a proposal whose window has closed, or
            # whose machine has drifted, must not come back with a button the
            # service would refuse.
            return approvals.decidable()

        async def get(action_id: str) -> HostActionRecord | None:
            try:
                return approvals.get(action_id)
            except UnknownAction:
                return None

        async def approve(
            action_id: str, chat_id: int, acknowledge: str
        ) -> ApprovalOutcome:
            refused = _refuse_unallowed(chat_id)
            if refused is not None:
                return refused
            try:
                record, _run_id = await approvals.approve(
                    action_id, actor=_actor(chat_id), acknowledge=acknowledge
                )
            except UnknownAction:
                return ApprovalOutcome(ok=False, message="no such host action")
            except (
                ConfirmationRequired,
                AlreadyDecided,
                ProposalExpired,
                HostdUnavailable,
                HostdError,
            ) as exc:
                return ApprovalOutcome(ok=False, message=str(exc))
            return ApprovalOutcome(
                ok=True,
                message=(
                    f"approved {record.proposal.summary} - applying it now; the "
                    "result follows"
                ),
                record=record,
            )

        async def deny(
            action_id: str, chat_id: int, reason: str
        ) -> ApprovalOutcome:
            refused = _refuse_unallowed(chat_id)
            if refused is not None:
                return refused
            # "-" is how the prompt offers "no reason", and an empty reason is
            # recorded as exactly that rather than as the literal dash.
            cleaned = "" if reason.strip() == "-" else reason.strip()
            try:
                record = await approvals.deny(
                    action_id, actor=_actor(chat_id), reason=cleaned
                )
            except UnknownAction:
                return ApprovalOutcome(ok=False, message="no such host action")
            except (AlreadyDecided, HostdUnavailable, HostdError) as exc:
                return ApprovalOutcome(ok=False, message=str(exc))
            told = (
                " The agent that asked has been told why."
                if cleaned and record.proposal.requester.agent
                else ""
            )
            return ApprovalOutcome(
                ok=True,
                message=f"denied {record.proposal.summary}.{told}",
                record=record,
            )

        return ApprovalOps(pending=pending, get=get, approve=approve, deny=deny)

    # Built whether or not a bot is running, and exposed: a test can then drive the
    # REAL decision path (and the real allowlist refusal) without starting a poll
    # loop against a stubbed Bot API, and the production wiring can be asserted
    # rather than assumed.
    app.state.telegram_approval_ops = _build_telegram_approval_ops()

    def _build_telegram_settings_ops() -> SettingsOps:
        """The read-only providers behind the bot's `/settings` and `/stats`
        commands, wired to the SAME in-process readers the web settings endpoints
        use (agent_health, read_usage, the orchestrator tool catalog, the host
        collector) - orchestrator-scoped, no self-HTTP."""

        async def info() -> OrchestratorInfo:
            orchestrator = agents.get(ORCHESTRATOR_ID)
            auth = auth_mode_for_backend(settings, orchestrator.backend)
            return OrchestratorInfo(
                backend=str(orchestrator.backend),
                model=orchestrator.model,
                auth_mode=str(auth) if auth is not None else None,
                enabled=settings.agent_enabled,
                permission_mode=str(settings.agent_permission_mode),
            )

        async def health() -> AgentHealth:
            _ensure_den_path(settings)  # so the in-process den probe sees the den
            return await agent_health(settings, is_orchestrator=True)

        async def usage() -> UsageQuota | None:
            if not settings.agent_enabled:
                return None
            # read_usage rglobs + parses every rollout: off-loop so a box with
            # many rollouts cannot stall the bot's poll loop (R1.1).
            return await asyncio.to_thread(read_usage, resolve_codex_home(settings))

        async def tools() -> list[AgentTool]:
            return await _tools_for_servers(_mcp_servers_for_audience(ORCHESTRATOR_ID))

        async def stats() -> HostStats:
            # collector.sample() is synchronous psutil I/O: off-loop (R1.1).
            return await asyncio.to_thread(collector.sample)

        return SettingsOps(
            info=info, health=health, usage=usage, tools=tools, stats=stats
        )

    def _start_telegram_bot() -> "asyncio.Task[None] | None":
        """Launch the in-process Telegram bot when a token is configured.

        The bot drives the orchestrator through the SAME internal turn path as
        the landing chat (`_launch_agent_turn` + `_drain_turn`) via injected
        callbacks - no self-HTTP. Returns the poll-loop task (the lifespan
        cancels it on shutdown), or None when no token is set. The bot and task
        are exposed on `app.state` for tests.
        """
        token = settings.telegram_bot_token
        if not token:
            app.state.telegram_bot = None
            app.state.telegram_task = None
            return None

        on_message, on_reset, on_cancel = build_telegram_callbacks(
            settings,
            agents,
            supervisor,
            _launch_agent_turn,
            lambda agent_id: agent_runs.get(agent_id),
        )
        bot = TelegramBot(
            token,
            settings.telegram_allowed_chat_ids,
            on_message,
            on_reset,
            on_cancel,
            settings_ops=_build_telegram_settings_ops(),
            approval_ops=app.state.telegram_approval_ops,
            stream=settings.telegram_stream,
        )
        task = asyncio.create_task(bot.run())
        app.state.telegram_bot = bot
        app.state.telegram_task = task
        return task

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
        if req.parent_session_id:
            # Stamp the child with the orchestrator chat that spawned it (part 3),
            # so a later request_input routes back to that chat.
            agents.record_spawn_parent(agent_id, ORCHESTRATOR_ID, req.parent_session_id)
        run_id, _bus = _launch_agent_turn(agent, project, goal)
        # Report the supervisor's actual state (usually "queued" until a slot is
        # free), not an assumed "running".
        started = supervisor.status(run_id)
        return RunStarted(
            agent_id=agent_id, state=started.state if started is not None else "running"
        )

    @app.post("/api/agents/{agent_id}/cancel")
    async def cancel_agent_run(agent_id: str) -> CancelResult:
        """Cancel the agent's in-flight run (the chat stop button, or the
        orchestrator's ``cancel_agent`` tool). Truly aborts the backend turn -
        the supervisor cancels the run task, whose drain aclose()s the backend
        stream so its cleanup runs (e.g. the Claude subprocess is killed). The
        persist callback then records a CANCELLED terminal outcome. Works for the
        orchestrator too (it is an agent in ``agent_runs`` keyed ORCHESTRATOR_ID).
        404 unknown agent, or 404 when the agent has no active run (mirroring
        ``/events``). Async: cancelling a task touches the running loop.

        A concurrent turn is refused elsewhere (409), so at most one run is live
        per agent - the single ``agent_runs[agent_id]`` entry is the one to stop.
        """
        _require_agent(agent_id)
        run_id = agent_runs.get(agent_id)
        if run_id is None or not supervisor.cancel(run_id):
            raise HTTPException(status_code=404, detail="no active run for this agent")
        return CancelResult(agent_id=agent_id, cancelled=True)

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
        # Expose the in-flight prompt only while the run is live, steering stripped
        # the same way read_transcript strips it, so a mid-turn reattach renders a
        # user bubble that matches the post-reload transcript (and its dedup guard).
        if (
            run_state is not None
            and run_state.state in (RunPhase.QUEUED, RunPhase.RUNNING)
            and run_state.prompt is not None
        ):
            result.prompt = strip_steering(run_state.prompt).strip() or None
        return result

    def _relay_bus_sse(
        bus: EventBus[_SseEventT], after_seq: int = 0
    ) -> StreamingResponse:
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

        return _relay_bus_sse(bus, _last_event_id(http_request))

    @app.post("/api/agents/{agent_id}/chat")
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
        decides, and the decision resumes the agent itself
        (``tasks/20260729-125040/DECISION.md`` section 5). The operator's own
        session may message it: reading its own chat is not deciding.

        LIVE, not merely BLOCKED: once the proposal is decided or its window has
        closed there is nothing for the orchestrator to interfere with, and refusing
        anyway would leave the agent unreachable for good (review round 1, R1.1).
        """
        agent = _require_agent(agent_id)
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        live = approvals.live_for_agent(agent_id)
        if live is not None and _caller_is_agent(request):
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
        project = _require_agent_project(agent)
        if req.parent_session_id:
            # Stamp the child with the orchestrator chat that sent this turn
            # (part 3), so a later request_input routes back to that chat.
            agents.record_spawn_parent(agent_id, ORCHESTRATOR_ID, req.parent_session_id)
        _run_id, bus = _launch_agent_turn(agent, project, message)
        return _relay_bus_sse(bus)

    @app.post("/api/agents/{agent_id}/request_input")
    def agent_request_input(
        agent_id: str, req: AgentRequestInput
    ) -> RequestInputResult:
        """A sub-agent signals it is blocked and needs a decision (BC2). Records a
        WAITING outcome carrying the question, keyed to the agent's CURRENT run so
        the turn-end completion preserves it (see ``AgentStore.request_input`` /
        ``mark_finished``); returns immediately - the agent ends its turn and the
        orchestrator answers later by resuming. 404 unknown agent (incl. the
        orchestrator, which is not a sub-agent), 422 empty question."""
        agent = _require_agent(agent_id)
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        try:
            outcome = agents.request_input(
                agent_id,
                question,
                run_id=agent_runs.get(agent_id, ""),
                session_id=agent.session_id,
            )
        except AgentNotFound as exc:
            # The orchestrator resolves via _require_agent but is not a sub-agent
            # (no agents.json row), so request_input rejects it - surface as 404.
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return RequestInputResult(agent_id=agent_id, state=outcome.state)

    @app.post("/api/agents/{agent_id}/report_back")
    def agent_report_back(agent_id: str, req: AgentReportBack) -> ReportBackResult:
        """A sub-agent signals it has FINISHED its task and hands back a result.
        Records a REPORTED outcome carrying the summary, keyed to the agent's
        CURRENT run so the turn-end completion preserves it (see
        ``AgentStore.report_back`` / ``mark_finished``); returns immediately - the
        agent ends its turn and the orchestrator is woken / sees it in
        `/api/agents/pending`, reads the report and acknowledges (no resume). 404
        unknown agent (incl. the orchestrator, which is not a sub-agent), 422 empty
        summary."""
        agent = _require_agent(agent_id)
        summary = req.summary.strip()
        if not summary:
            raise HTTPException(status_code=422, detail="summary must not be empty")
        try:
            outcome = agents.report_back(
                agent_id,
                summary,
                run_id=agent_runs.get(agent_id, ""),
                session_id=agent.session_id,
            )
        except AgentNotFound as exc:
            # The orchestrator resolves via _require_agent but is not a sub-agent
            # (no agents.json row), so report_back rejects it - surface as 404.
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return ReportBackResult(agent_id=agent_id, state=outcome.state)

    @app.post("/api/agents/{agent_id}/acknowledge")
    def agent_acknowledge(agent_id: str) -> AcknowledgeResult:
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
        if approvals.live_for_agent(agent_id) is not None:
            return AcknowledgeResult(agent_id=agent_id, acknowledged=False)
        return AcknowledgeResult(
            agent_id=agent_id, acknowledged=agents.acknowledge(agent_id)
        )

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
        settings page shares this endpoint. 404 unknown; never raises otherwise.

        The MCP health rows are scoped to THIS agent's audience: the orchestrator
        gets its scufris + den servers, a sub-agent its callback server, a backend
        with no scufris MCP a single "none" row."""
        agent = _require_agent(agent_id)
        _ensure_den_path(settings)  # so the in-process den probe sees the den
        return await agent_health(
            settings,
            backend=agent.backend,
            is_orchestrator=agent.id == ORCHESTRATOR_ID,
            agent_id=agent.id,
            has_scufris_mcp=_agent_has_scufris_mcp(agent),
        )

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

    def _as_agent_tool(t: Any, server: str, disabled: set[str]) -> AgentTool:
        schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
        props = schema.get("properties")
        args = list(props) if isinstance(props, dict) else []
        return AgentTool(
            name=t.name,
            description=t.description or "",
            server=server,
            args=args,
            parameters=_tool_parameters(t.inputSchema),
            enabled=t.name not in disabled,
        )

    def _agent_has_scufris_mcp(agent: AgentRecord) -> bool:
        # Which backends actually wire the scufris MCP servers into an agent's turn:
        # codex via `-c mcp_servers.<id>.*` (agent._mcp_overrides) and claude via
        # `--mcp-config` (backends._scufris_claude_args) - both from the shared
        # scufris_mcp_servers core, so both register the same servers per audience.
        # opencode/mock have no scufris wiring, so they deliver NO scufris tools and
        # their real tool surface is empty regardless of audience.
        backend = canonical_backend(agent.backend)
        return backend in ("codex", "claude")

    def _mcp_servers_for_audience(agent_id: str) -> list[tuple[str, Any]]:
        """The in-process ``(server_id, FastMCP)`` pairs for an agent's audience:
        the orchestrator's ``scufris`` + ``den``, the host agent's ``host`` +
        ``agent``, or a regular sub-agent's ``agent`` server (from
        ``mcp_health.servers_for_audience``, which mirrors what a real turn
        registers)."""
        from .mcp_health import servers_for_audience

        return servers_for_audience(agent_id == ORCHESTRATOR_ID, agent_id)

    async def _tools_for_servers(servers: list[tuple[str, Any]]) -> list[AgentTool]:
        """Aggregate the tools of the given in-process ``(server_id, FastMCP)`` pairs,
        each tool tagged with its real server id and its enabled flag from the
        operator disabled-tool set. Mirrors what a real turn registers, so the
        read-only listing matches what the audience actually gets."""
        disabled = set(settings.disabled_tools)
        out: list[AgentTool] = []
        for server_id, mcp in servers:
            for t in await mcp.list_tools():
                out.append(_as_agent_tool(t, server_id, disabled))
        return out

    async def _probe_servers(
        servers: list[tuple[str, Any]],
    ) -> list[McpServerHealth]:
        """Live-probe each server (``mcp_health.probe_server``) into an
        ``McpServerHealth`` for the settings "MCP tools" section: the server's
        status/detail plus its tools, each tool's ``available`` flag set from the
        server's probe verdict and ``enabled`` from the operator disabled-tool set.
        The den path must already be bridged (call ``_ensure_den_path`` first) so the
        den readiness check sees it."""
        from .mcp_health import probe_server

        disabled = set(settings.disabled_tools)
        out: list[McpServerHealth] = []
        for server_id, mcp in servers:
            status, detail, available, tools = await probe_server(
                server_id, mcp, disabled
            )
            agent_tools: list[AgentTool] = []
            for t in tools:
                at = _as_agent_tool(t, server_id, disabled)
                at.available = available
                agent_tools.append(at)
            out.append(
                McpServerHealth(
                    id=server_id, status=status, detail=detail, tools=agent_tools
                )
            )
        return out

    @app.get("/api/agent/tools")
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
        return await _tools_for_servers(_mcp_servers_for_audience(ORCHESTRATOR_ID))

    @app.get("/api/agents/{agent_id}/tools")
    async def get_agent_scoped_tools(agent_id: str) -> list[AgentTool]:
        """The scufris MCP tools THIS agent can actually call in its turns -
        AUDIENCE- and BACKEND-scoped, read-only. A codex or claude sub-agent gets
        only the ``agent`` callback server (request_input/report_back); the
        orchestrator gets its ``scufris`` + ``den`` servers; an agent whose backend
        does not wire the scufris MCP (opencode/mock, today) gets NONE. This is what
        the agent's settings page shows, so the display matches what the agent
        really has - unlike the orchestrator-console ``/api/agent/tools``. 404
        unknown agent."""
        agent = _require_agent(agent_id)
        if not _agent_has_scufris_mcp(agent):
            return []
        return await _tools_for_servers(_mcp_servers_for_audience(agent.id))

    @app.get("/api/agent/mcp")
    async def get_agent_mcp() -> list[McpServerHealth]:
        """Live per-server health for the operator console's "MCP tools" section:
        the orchestrator's ``scufris`` + ``den`` servers, each with a probe status
        (green/amber/red) and its tools carrying per-tool enabled/available flags.
        For a SPECIFIC agent's servers, see ``GET /api/agents/{id}/mcp``."""
        _ensure_den_path(settings)
        return await _probe_servers(_mcp_servers_for_audience(ORCHESTRATOR_ID))

    @app.get("/api/agents/{agent_id}/mcp")
    async def get_agent_scoped_mcp(agent_id: str) -> list[McpServerHealth]:
        """Live per-server health for THIS agent's audience: the orchestrator's
        ``scufris`` + ``den``, or a sub-agent's ``agent`` callback server. Empty when
        the agent's backend wires no scufris MCP (opencode/mock). 404 unknown
        agent."""
        agent = _require_agent(agent_id)
        if not _agent_has_scufris_mcp(agent):
            return []
        _ensure_den_path(settings)
        return await _probe_servers(_mcp_servers_for_audience(agent.id))

    @app.get("/api/agents/{agent_id}/capabilities")
    def get_agent_capabilities(agent_id: str) -> ProjectCapabilities:
        """The read-only skills + custom tools THIS agent's PROJECT defines in its
        working tree - its ``.claude/skills`` / ``.mcp.json`` (claude) or
        ``.codex`` equivalents (codex), discovered PROVIDER-aware from the agent's
        backend. What the settings page surfaces so the operator can see the
        recipes and tools an agent can be steered toward. The orchestrator (and any
        project-less agent) has no project tree -> an empty set. 404 unknown agent;
        nothing here is writable or executed."""
        agent = _require_agent(agent_id)
        project = _require_agent_project(agent)
        if project is None:
            return ProjectCapabilities()
        return read_project_capabilities(project.cwd, canonical_backend(agent.backend))

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

        if name in set(settings.disabled_tools):
            raise HTTPException(status_code=403, detail=f"tool {name!r} is disabled")
        # The console runs the orchestrator's tools, now spread across two servers
        # (scufris + den); find the one that owns this tool.
        target = None
        for _server_id, m in _mcp_servers_for_audience(ORCHESTRATOR_ID):
            if any(t.name == name for t in await m.list_tools()):
                target = m
                break
        if target is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {name!r}")
        # This runs the tool IN THIS process, so bridge the den path from settings
        # into the env the journal_* tools read (the agent path injects it into the
        # MCP subprocess instead; see _ensure_den_path). No-op for non-journal tools.
        _ensure_den_path(settings)
        # This tool may call THIS server's own API (`mcp_common._api_call`), which
        # is now gated. An MCP subprocess gets the machine token through its
        # injected env; an in-process run gets it here. A ContextVar rather than
        # os.environ so a second app in the same process cannot clobber it, and so
        # nothing else in the process picks it up ambiently; `asyncio.to_thread`
        # copies the context, so it survives the off-loop hop below.
        api_token_var.set(app.state.api_token)
        try:
            # Run the tool OFF the event loop. FastMCP calls a SYNC tool inline
            # (`return fn(...)`), so an HTTP-backed tool's BLOCKING httpx call would
            # otherwise run on the loop - and when the base points at THIS server
            # (the common case now, see `_ensure_api_base`), that self-loopback
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

    @app.get("/api/agent/health")
    async def get_agent_health() -> AgentHealth:
        """Read-only diagnostics for the operator console (never raises). The MCP
        rows are the orchestrator's scufris + den servers."""
        _ensure_den_path(settings)  # so the in-process den probe sees the den
        return await agent_health(settings, is_orchestrator=True)

    @app.get("/api/agent/sessions")
    def get_sessions() -> SessionsResponse:
        """List the orchestrator's own sessions (to switch between) + the current
        one. Driven by the ownership registry, not a provider disk scan: only
        sessions the registry attributes to the orchestrator appear, so a
        sub-agent's chat can never leak in (part 1). Each id is hydrated through
        the orchestrator's backend, so this works for codex/claude/opencode
        alike."""
        if not settings.agent_enabled:
            return SessionsResponse(sessions=[], current=None)
        backend = get_backend(agents.get(ORCHESTRATOR_ID).backend)
        infos = [
            info
            for sid in agents.orchestrator_sessions()
            if (info := session_info(backend, settings, sid)) is not None
        ]

        def _activity(info: SessionInfo) -> float:
            # Newest first by last activity; fall back to start time, then 0.
            when = info.updated_at or info.started_at
            return when.timestamp() if when is not None else 0.0

        infos.sort(key=_activity, reverse=True)
        return SessionsResponse(
            sessions=infos,
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
        backend = get_backend(agents.get(ORCHESTRATOR_ID).backend)
        messages = backend.read_transcript(settings, request.source_id)
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
        """The current session's context snapshot (window + token usage + counts),
        read through the orchestrator's backend so it works for codex/claude/
        opencode (codex keeps the rich token breakdown; others map read_status)."""
        if not settings.agent_enabled:
            return None
        backend = get_backend(agents.get(ORCHESTRATOR_ID).backend)
        return backend.read_context(settings, agents.orchestrator_session_id())

    @app.get("/api/agent/session/{session_id}")
    def get_session_transcript(session_id: str) -> TranscriptResponse:
        """A session's past messages, so switching to it re-renders its history -
        read through the orchestrator's backend (codex/claude/opencode)."""
        if not settings.agent_enabled:
            return TranscriptResponse(messages=[])
        backend = get_backend(agents.get(ORCHESTRATOR_ID).backend)
        return TranscriptResponse(
            messages=backend.read_transcript(settings, session_id)
        )

    @app.delete("/api/agent/session/{session_id}")
    async def delete_agent_session(session_id: str) -> DeleteResult:
        """Delete a session: remove its provider-side record via the orchestrator's
        backend (codex rollout / claude file / opencode daemon) AND forget it from
        the switcher history, so it leaves the list. ``forget`` also clears the
        current pointer when it was the active session; a backend with no provider
        delete still drops the session from the list."""
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with supervisor.serialized(ORCHESTRATOR_ID):
            backend = get_backend(agents.get(ORCHESTRATOR_ID).backend)
            deleted = await backend.delete_session(settings, session_id)
            agents.forget_orchestrator_session(session_id)
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
        return _relay_bus_sse(bus, _last_event_id(http_request))

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


def _ensure_api_base(settings: Settings) -> str:
    """Default ``SCUFRIS_API_BASE`` to THIS dashboard's own base, so an in-process
    tool run (the operator console's ``/api/agent/tools/{name}/run``) loops back to
    this server rather than ``mcp_server._api_base``'s hardcoded ``:8000`` default -
    which, on a non-8000 port, silently hits a different (often stale) instance.

    ``setdefault`` so an explicit operator override wins (a non-loopback
    deployment). ``127.0.0.1`` rather than ``settings.host`` because the host may
    be ``0.0.0.0`` (bind-all), which is not a connectable address. Returns the
    effective base."""
    return os.environ.setdefault(
        "SCUFRIS_API_BASE", f"http://127.0.0.1:{settings.port}"
    )


def _ensure_den_path(settings: Settings) -> None:
    """Bridge ``settings.den_path`` into ``SCUFRIS_DEN_PATH`` for an IN-PROCESS tool
    run (the operator console's ``/api/agent/tools/{name}/run``), so the ``journal_*``
    tools resolve the den the same way they do in an agent turn.

    The journal tools read ``SCUFRIS_DEN_PATH`` from the environment
    (``mcp_server._den_path``), which the agent path injects into the MCP SUBPROCESS
    env. The console runs the tool in THIS process instead, and pydantic loads
    ``den_path`` from ``.env`` into the ``Settings`` object WITHOUT exporting it to
    ``os.environ`` - so without this bridge the console sees an unset var and reports
    "not configured". Mirrors ``_ensure_api_base``. ``setdefault`` so an explicit env
    (the deployed service sets ``SCUFRIS_DEN_PATH`` directly) wins; a no-op when
    ``den_path`` is unset (the tools stay correctly inert). Isolation is unaffected:
    a sub-agent cannot call ``journal_*`` at all (the ``den`` server is never
    registered on a sub-agent turn), so a subprocess inheriting the var is moot."""
    if settings.den_path is not None:
        os.environ.setdefault("SCUFRIS_DEN_PATH", str(settings.den_path))


def run_server(settings: Settings | None = None) -> None:
    """Launch the dashboard app with uvicorn."""
    import uvicorn

    settings = settings or Settings()
    _ensure_api_base(settings)
    # Un-forced: the CLI has usually already configured (honouring --debug); a
    # direct run_server() call configures from the setting instead.
    configure_logging(settings.log_level)
    # Check the auth posture BEFORE announcing a start: create_app would raise
    # anyway, but only after the log line has claimed the server is coming up.
    validate_auth_config(settings)
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
