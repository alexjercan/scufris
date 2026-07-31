"""The codex adapter: the ``app_server`` runner plus the rollout readers.

``stream`` always uses ``codex app-server`` - the turn-level ``exec`` runner is no
longer a per-agent choice. Everything read back comes from the rollout JSONL
codex already writes, through ``sessions``.
"""

from __future__ import annotations

from typing import AsyncIterator

from ..agent import StreamEvent, _stream_app_server
from ..config import Settings
from ..logsetup import truncate
from ..reasoning_store import ReasoningStore
from ..sessions import (
    SessionContext,
    TranscriptMessage,
    merge_reasoning,
    read_context,
    read_transcript,
    resolve_codex_home,
    rollout_mtime,
)
from ..sessions import (
    delete_session as codex_delete_session,
)
from .base import _LAST_MESSAGE_PREVIEW, BackendStatus

#: Permission mode -> codex `--sandbox` value (verified live via `--help`).
_CODEX_SANDBOX = {
    "manual": "read-only",
    "edit": "workspace-write",
    "auto": "danger-full-access",
}


def _codex_sandbox_for(mode: str) -> str:
    return _CODEX_SANDBOX.get(mode, "read-only")


class CodexBackend:
    """The "codex" backend: codex's ``app_server`` runner (token streaming). The
    turn-level ``exec`` runner is no longer a per-agent choice; ``stream``
    always uses app_server. ``name`` is the friendly id "codex";
    ``read_status`` reads the rollout via ``sessions``."""

    def __init__(self) -> None:
        self.name: str = "codex"

    async def stream(
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
        sandbox = _codex_sandbox_for(permission_mode)
        async for event in _stream_app_server(
            settings,
            prompt,
            session_id,
            image_paths,
            cwd=cwd,
            sandbox=sandbox,
            is_orchestrator=is_orchestrator,
            agent_id=agent_id,
        ):
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

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        messages = read_transcript(resolve_codex_home(settings), session_id)
        # Reasoning is not on disk (encrypted blob), so re-hydrate the "thinking"
        # spoilers from scufris's own sidecar - merged here (not in the pure
        # rollout reader) because the sidecar lives under state_dir, not codex_home.
        merge_reasoning(messages, ReasoningStore(settings).read(session_id))
        return messages

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # The rich rollout reader (keeps cached/reasoning/total + window).
        return read_context(resolve_codex_home(settings), session_id)

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        return codex_delete_session(resolve_codex_home(settings), session_id)
