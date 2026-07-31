"""The ``AgentBackend`` interface: one seam over codex, claude, opencode and mock.

The orchestrator, the run supervisor, the dashboard and the store all speak to an
agent through this interface, so nothing above it branches on which backend an
agent uses. A backend does two things:

- ``stream(...)`` runs one turn scoped to the agent's project ``cwd``, resuming a
  ``session_id`` when given, and yields normalized ``StreamEvent``s (the same
  events the event bus fans out).
- ``read_status(...)`` returns a READ-ONLY snapshot of a session's progress
  derived from its durable log (for codex, the rollout JSONL) - the "what is this
  agent doing" half. The live run-state (queued/running/done) comes from the
  Supervisor and is merged with this above.

Four adapters sit behind the SAME interface, which is what proves the interface
is not accidentally codex-shaped. The protocol lives in exactly one module
(``base``), each adapter owns its own permission/sandbox/tool-mode map, and
``get_backend`` below is the only place that knows every adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import Settings, canonical_backend
from ..sessions import SessionInfo
from .base import AgentBackend, BackendStatus
from .claude import (
    ClaudeBackend,
    _claude_stream_args,
    parse_claude_stream,
    parse_claude_transcript,
    resolve_claude_home,
)
from .codex import CodexBackend
from .mock import MockBackend
from .opencode import OpenCodeBackend, _opencode_tools_for, parse_opencode_transcript

# Cap a switcher title to the same length codex's own lister used
# (`sessions.rollout._TITLE_MAX`), so titles do not vary by which path built them.
_SESSION_TITLE_MAX = 80


def session_info(
    backend: AgentBackend, settings: Settings, session_id: str
) -> SessionInfo | None:
    """A ``SessionInfo`` for the orchestrator switcher, hydrated backend-agnostically
    from the session's OWN store: title = its first user message,
    ``started_at`` = that message's timestamp, ``updated_at`` = the backend status
    snapshot's mtime. Returns None when the id resolves to no readable session, so
    a stale/foreign id drops out of the list instead of showing as "(untitled)".

    ``git_branch``/``cwd`` are left unset: they are codex rollout-meta niceties
    the generic (transcript + status) path does not parse, and the switcher UI
    does not depend on them. Reads the transcript and a status snapshot, so it is
    heavier than a head-only scan - fine for a short switcher list."""
    messages = backend.read_transcript(settings, session_id)
    status = backend.read_status(settings, session_id)
    if not messages and status is None:
        return None
    title = ""
    started_at: datetime | None = None
    for message in messages:
        if message.role == "user" and message.text.strip():
            title = message.text.strip()
            started_at = message.ts
            break
    updated_at: datetime | None = None
    if status is not None and status.updated_at is not None:
        updated_at = datetime.fromtimestamp(status.updated_at, timezone.utc)
    return SessionInfo(
        id=session_id,
        title=(title or "(untitled)")[:_SESSION_TITLE_MAX],
        started_at=started_at,
        updated_at=updated_at,
    )


def get_backend(name: str) -> AgentBackend:
    """Resolve a backend by (possibly legacy) name; unknown raises.

    Legacy codex modes (`app_server`/`exec`) and `codex` all resolve to the codex
    backend (app_server runner). `mock` always RESOLVES (an already-persisted mock
    agent must still run); whether a mock agent may be CREATED is gated separately
    by `enable_mock_backend` in the store.
    """
    canonical = canonical_backend(name)
    if canonical == "codex":
        return CodexBackend()
    if canonical == "claude":
        return ClaudeBackend()
    if canonical == "opencode":
        return OpenCodeBackend()
    if canonical == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend: {name!r}")


__all__ = [
    "AgentBackend",
    "BackendStatus",
    "ClaudeBackend",
    "CodexBackend",
    "MockBackend",
    "OpenCodeBackend",
    "get_backend",
    "parse_claude_stream",
    "parse_claude_transcript",
    "parse_opencode_transcript",
    "resolve_claude_home",
    "session_info",
    # Internal to scufris, but part of this package's import surface: the tests
    # exercise the claude arg builder and the opencode tool map directly.
    "_claude_stream_args",
    "_opencode_tools_for",
]
