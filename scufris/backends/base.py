"""The ``AgentBackend`` interface and the status snapshot every adapter returns.

This is the seam itself: the orchestrator, the run supervisor, the dashboard and
the store all speak to an agent through the protocol here, so nothing above it
branches on which backend an agent uses. Keeping the contract in exactly one
module - with no adapter in it - is what stops it drifting into a codex-shaped
interface with three special cases bolted on.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel

from ..agent import StreamEvent
from ..config import Settings
from ..sessions import SessionContext, TranscriptMessage

#: How much of the last assistant message to keep in a status snapshot.
_LAST_MESSAGE_PREVIEW = 280


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


def _context_from_status(status: BackendStatus | None) -> SessionContext | None:
    """Map a ``BackendStatus`` onto a ``SessionContext`` for backends that expose no
    richer per-session context breakdown than their status snapshot (claude,
    opencode). The codex-only cached/reasoning/total token axes stay 0; window is
    whatever the status reports (0 when the backend does not surface one)."""
    if status is None:
        return None
    return SessionContext(
        session_id=status.session_id,
        context_window=status.context_window,
        input_tokens=status.input_tokens,
        output_tokens=status.output_tokens,
        turn_count=status.turns,
        tool_call_count=status.tool_calls,
    )


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
        permission_mode: str = "manual",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn in ``cwd``, resuming ``session_id`` if given; yield events.

        ``permission_mode`` is the agent's write posture (manual|edit|auto),
        mapped per backend to codex's ``--sandbox`` / claude's
        ``--permission-mode``. Default ``manual`` = read-only.

        ``is_orchestrator`` marks the turn as the landing orchestrator's, which
        selects the orchestrator role of the scufris MCP server and its steering
        (codex backend). ``agent_id`` is the caller's own id, passed to a regular
        (non-orchestrator) agent's scufris server so its ``request_input`` callback
        can address itself. Both are a no-op for backends without MCP wiring
        (claude, mock).
        """
        ...

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        """A read-only progress snapshot for ``session_id``, or None if unreadable."""
        ...

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        """The session's past messages (for rebuilding the chat), oldest-first;
        empty when ``session_id`` is unset or the session cannot be read."""
        ...

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        """The session's context snapshot (window + token usage + counts), or None
        when unset/unreadable. codex reads the rich rollout breakdown; other
        backends map what their status snapshot exposes (window 0 when the backend
        does not report one)."""
        ...

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        """Delete the session's provider-side record, returning True if removed.
        Codex unlinks the rollout, claude the transcript file, opencode calls the
        daemon; a backend with no provider delete returns False. Never raises - a
        False just means the registry forget (caller side) is the only cleanup."""
        ...
