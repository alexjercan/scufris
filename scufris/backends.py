"""The ``AgentBackend`` interface: one seam over codex and (later, A2b) claude.

The orchestrator, the run supervisor, the dashboard and the store all speak to an
agent through this interface, so nothing above it branches on which backend an
agent uses (spike tasks/20260720-221748 decisions 1 and 4). A backend does two
things:

- ``stream(...)`` runs one turn scoped to the agent's project ``cwd``, resuming a
  ``session_id`` when given, and yields normalized ``StreamEvent``s (the same
  events the A0 event bus fans out).
- ``read_status(...)`` returns a READ-ONLY snapshot of a session's progress
  derived from its durable log (for codex, the rollout JSONL) - the "what is this
  agent doing" half. The live run-state (queued/running/done) comes from the A0
  Supervisor and is merged with this in A3/A5.

A2 ships ``CodexBackend`` (exec + app_server) and ``MockBackend``; A2b adds a
``claude`` backend behind the SAME interface, which is what proves the interface
is not accidentally codex-shaped.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from .agent import (
    AgentReply,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    _stream_app_server,
    _stream_codex_exec,
)
from .config import Settings
from .logsetup import truncate
from .sessions import (
    read_context,
    read_transcript,
    resolve_codex_home,
    rollout_mtime,
)

logger = logging.getLogger(__name__)

# How much of the last assistant message to keep in a status snapshot.
_LAST_MESSAGE_PREVIEW = 280

CodexMode = Literal["exec", "app_server"]


class BackendStatus(BaseModel):
    """A read-only snapshot of one agent session's progress, normalized across
    backends. Derived from the durable log, not the live run - ``state`` is left
    to the Supervisor and merged in later."""

    session_id: str
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_window: int = 0
    last_message: str | None = None
    updated_at: float | None = None


@runtime_checkable
class AgentBackend(Protocol):
    """What the orchestrator/supervisor depend on; implementations are swappable."""

    name: str

    def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn in ``cwd``, resuming ``session_id`` if given; yield events."""
        ...

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        """A read-only progress snapshot for ``session_id``, or None if unreadable."""
        ...


class CodexBackend:
    """codex behind the interface, in ``exec`` (turn-level) or ``app_server``
    (token-streaming) mode. ``stream`` delegates to the agent.py runners (with the
    A0 ``cwd`` seam); ``read_status`` reads the rollout via sessions.py."""

    def __init__(self, mode: CodexMode) -> None:
        self._mode: CodexMode = mode
        self.name: str = mode

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        runner = (
            _stream_app_server if self._mode == "app_server" else _stream_codex_exec
        )
        async for event in runner(settings, prompt, session_id, image_paths, cwd=cwd):
            yield event

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        home = resolve_codex_home(settings)
        ctx = read_context(home, session_id)
        if ctx is None:
            return None
        last_message: str | None = None
        for msg in reversed(read_transcript(home, session_id)):
            if msg.role == "assistant" and msg.text.strip():
                last_message = truncate(msg.text.strip(), _LAST_MESSAGE_PREVIEW)
                break
        return BackendStatus(
            session_id=session_id,
            turns=ctx.turn_count,
            tool_calls=ctx.tool_call_count,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            context_window=ctx.context_window,
            last_message=last_message,
            updated_at=rollout_mtime(home, session_id),
        )


class MockBackend:
    """An in-process backend for tests/offline demos - no codex, no network."""

    name: str = "mock"

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta=f"[mock] {prompt}")
        yield StreamDone(
            reply=AgentReply(text=f"[mock reply] {prompt}"),
            session_id=session_id or "mock-session",
        )

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        return BackendStatus(
            session_id=session_id, turns=1, last_message="[mock] running"
        )


def get_backend(name: str) -> AgentBackend:
    """Resolve a backend by name. A2b adds ``"claude"``; unknown raises."""
    if name in ("exec", "app_server"):
        return CodexBackend(name)  # type: ignore[arg-type]  # narrowed by the check
    if name == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend: {name!r}")
