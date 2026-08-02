"""The mock adapter: an in-process backend for tests and offline demos."""

from __future__ import annotations

from typing import AsyncIterator

from ..agent import AgentReply, StreamDone, StreamEvent, StreamTextDelta
from ..config import Settings
from ..sessions import (
    MemoryFootprint,
    SessionContext,
    TranscriptMessage,
    UsageQuota,
)
from .base import BackendStatus, Capability


class MockBackend:
    """An in-process backend for tests/offline demos - no codex, no network."""

    name: str = "mock"
    has_scufris_mcp: bool = False

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

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        # The in-process mock keeps no on-disk transcript.
        return []

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # The in-process mock keeps no context snapshot.
        return None

    def read_usage(self, settings: Settings) -> Capability[UsageQuota]:
        # No account and nothing on disk to measure.
        return Capability.unsupported()

    def read_memory_footprint(self, settings: Settings) -> Capability[MemoryFootprint]:
        return Capability.unsupported()

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        # Nothing on disk / no daemon to delete from.
        return False
