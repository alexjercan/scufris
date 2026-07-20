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
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from .agent import Agent, AgentHandle, AgentReply, AgentUnavailable, build_agent
from .config import SERVER_ID_RE, McpServerSpec, Settings
from .health import AgentHealth, agent_health
from .logsetup import configure_logging, new_request_id, set_request_id
from .metrics import Collector, HostStats, PsutilCollector
from .processes import ProcessCollector, ProcessList, PsutilProcessCollector
from .sessions import (
    SessionContext,
    SessionInfo,
    TranscriptMessage,
    UsageQuota,
    delete_session,
    format_fork_seed,
    list_sessions,
    read_context,
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

logger = logging.getLogger(__name__)


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
    auth_mode: str
    enabled: bool


class AgentTool(BaseModel):
    name: str
    description: str
    server: str = "scufris"  # the MCP server that exposes it
    args: list[str] = []  # parameter names, from the tool's input schema
    enabled: bool = True  # False when the operator disabled it (disabled_tools)


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
    auth_mode: str
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
    agent_backend: Literal["app_server", "exec", "mock"] | None = None
    agent_model: str | None = None
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


def create_app(
    collector: Collector | None = None,
    settings: Settings | None = None,
    agent: Agent | None = None,
    process_collector: ProcessCollector | None = None,
) -> FastAPI:
    settings = settings or Settings()
    collector = collector or PsutilCollector()
    # An injected agent (tests) is used as-is; otherwise wrap the built agent in
    # a handle so a live change to agent_enabled/agent_backend can rebuild it.
    handle: AgentHandle | None
    if agent is None:
        handle = AgentHandle(settings, build_agent)
        agent = handle
    else:
        handle = None
    process_collector = process_collector or PsutilProcessCollector()
    # Runtime-mutable settings: env base with persisted overrides layered on.
    # Mutations happen in place, so the closures below (and the agent) read the
    # new value live; a rebuild-class key notifies the handle to rebuild.
    store = SettingsStore(
        settings, on_change=(lambda _changed: handle.rebuild()) if handle else None
    )
    # Codex sessions are not concurrency-safe; serialize chat turns.
    chat_lock = asyncio.Lock()

    app = FastAPI(title="Scufris", description="Scuffed Jarvis host dashboard")

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
            auth_mode=settings.agent_auth_mode,
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
            auth_mode=settings.agent_auth_mode,
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
            if spec.id == "scufris" or not re.fullmatch(SERVER_ID_RE, spec.id):
                raise HTTPException(
                    status_code=422, detail=f"invalid MCP server id: {spec.id!r}"
                )
            if not spec.command.strip():
                raise HTTPException(
                    status_code=422, detail="MCP server command must not be empty"
                )
        updates = update.model_dump(exclude_none=True)
        try:
            store.apply(updates)
        except SettingsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (UnknownSettingKey, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
                    enabled=t.name not in disabled,
                )
            )
        return result

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
            current=agent.current_session_id(),
        )

    @app.post("/api/agent/session")
    async def post_session(action: SessionAction) -> CurrentSession:
        """Start a new session or switch to an existing one for the next turn."""
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with chat_lock:
            if action.action == "switch":
                if not action.session_id:
                    raise HTTPException(
                        status_code=422, detail="session_id required to switch"
                    )
                agent.switch_session(action.session_id)
            else:
                agent.new_session()
            return CurrentSession(current=agent.current_session_id())

    @app.post("/api/agent/session/fork")
    async def fork_session(request: ForkRequest) -> ForkResult:
        """Fork a conversation: start a new session seeded with the turns before
        the edited message plus the edited text, and run it as the first turn.

        codex-exec has no native branch, so the prior turns are pasted as context.
        """
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        async with chat_lock:
            home = resolve_codex_home(settings)
            messages = read_transcript(home, request.source_id)
            cut = max(0, request.message_index)
            seed = format_fork_seed(messages[:cut], request.text)
            agent.new_session()
            try:
                reply = await agent.chat(seed)
            except AgentUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return ForkResult(current=agent.current_session_id(), reply=reply)

    @app.get("/api/agent/context")
    def get_context() -> SessionContext | None:
        """The current session's context snapshot (window + token usage + counts)."""
        if not settings.agent_enabled:
            return None
        return read_context(resolve_codex_home(settings), agent.current_session_id())

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
        async with chat_lock:
            deleted = delete_session(resolve_codex_home(settings), session_id)
            if deleted and agent.current_session_id() == session_id:
                agent.new_session()
            return DeleteResult(deleted=deleted, current=agent.current_session_id())

    @app.get("/api/agent/usage")
    def get_usage() -> UsageQuota | None:
        """Account-wide usage/quota (the weekly rate-limit window)."""
        if not settings.agent_enabled:
            return None
        return read_usage(resolve_codex_home(settings))

    @app.post("/api/chat")
    async def post_chat(request: ChatRequest) -> AgentReply:
        """Send one message to the agent and return its reply (turn-based)."""
        async with chat_lock:
            try:
                return await agent.chat(request.message)
            except AgentUnavailable as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    async def post_chat_stream(request: ChatRequest) -> StreamingResponse:
        """Send one message and stream live turn progress as SSE.

        Each `data:` frame is a JSON stream event (`tool` as each MCP tool
        completes, then `done` with the reply, or `error`). The chat lock is held
        for the whole stream so turns stay serialized.
        """
        if not settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")

        async def events() -> AsyncIterator[str]:
            # A leading SSE comment (ignored by the client parser) flushes the
            # headers and primes the connection immediately, and its padding
            # pushes past any residual browser MIME-sniff buffer so the first
            # real tokens are not withheld. The model reasons for a few seconds
            # before the first token, so this also confirms the stream is open.
            yield f":{' ' * 2048}\n\n"
            tmpdir: str | None = None
            image_paths: list[str] | None = None
            if request.image is not None:
                try:
                    tmpdir, path = _write_image_to_temp(request.image)
                    image_paths = [path]
                except ValueError as exc:
                    payload = json.dumps({"kind": "error", "detail": str(exc)})
                    yield f"data: {payload}\n\n"
                    return
            try:
                async with chat_lock:
                    try:
                        async for event in agent.chat_stream(
                            request.message, image_paths=image_paths
                        ):
                            yield f"data: {event.model_dump_json()}\n\n"
                    except AgentUnavailable as exc:
                        payload = json.dumps({"kind": "error", "detail": str(exc)})
                        yield f"data: {payload}\n\n"
            finally:
                if tmpdir is not None:
                    shutil.rmtree(tmpdir, ignore_errors=True)

        # SSE-friendly headers so tokens reach the browser as they are yielded
        # rather than being withheld: `nosniff` stops Chrome buffering the first
        # ~1KB of a fetch ReadableStream for MIME sniffing (which lumps the first
        # tokens together); `no-cache`/`X-Accel-Buffering: no` defeat client and
        # reverse-proxy (nginx) response buffering.
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

    @app.post("/api/chat/reset")
    async def post_chat_reset() -> dict[str, bool]:
        """Start a fresh conversation (forget prior context)."""
        async with chat_lock:
            agent.reset()
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
