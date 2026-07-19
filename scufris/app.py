"""The Scufris FastAPI application.

Serves a read-only JSON stats API and, when built, the static dashboard bundle.
The stats collector is injected so tests can supply a fake; production uses the
psutil-backed collector.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent, AgentReply, AgentUnavailable, build_agent
from .config import Settings
from .metrics import Collector, HostStats, PsutilCollector
from .processes import ProcessCollector, ProcessList, PsutilProcessCollector
from .sessions import (
    SessionContext,
    SessionInfo,
    TranscriptMessage,
    UsageQuota,
    list_sessions,
    read_context,
    read_transcript,
    read_usage,
    resolve_codex_home,
)

logger = logging.getLogger(__name__)


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


class SessionsResponse(BaseModel):
    sessions: list[SessionInfo]
    current: str | None


class CurrentSession(BaseModel):
    current: str | None


class TranscriptResponse(BaseModel):
    messages: list[TranscriptMessage]


class SessionAction(BaseModel):
    action: Literal["new", "switch"]
    session_id: str | None = None


class ChatRequest(BaseModel):
    message: str


def create_app(
    collector: Collector | None = None,
    settings: Settings | None = None,
    agent: Agent | None = None,
    process_collector: ProcessCollector | None = None,
) -> FastAPI:
    settings = settings or Settings()
    collector = collector or PsutilCollector()
    agent = agent if agent is not None else build_agent(settings)
    process_collector = process_collector or PsutilProcessCollector()
    # Codex sessions are not concurrency-safe; serialize chat turns.
    chat_lock = asyncio.Lock()

    app = FastAPI(title="Scufris", description="Scuffed Jarvis host dashboard")

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

    @app.get("/api/agent/tools")
    async def get_agent_tools() -> list[AgentTool]:
        """The curated tools the agent can call (from the Scufris MCP server)."""
        from .mcp_server import mcp

        tools = await mcp.list_tools()
        return [AgentTool(name=t.name, description=t.description or "") for t in tools]

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
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
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

    logging.basicConfig(level=logging.INFO)
    settings = settings or Settings()
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)
