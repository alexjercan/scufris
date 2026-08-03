"""The orchestrator turn service: one conversation, three transports.

The landing chat, the Telegram bot and the wake bridge each used to repeat the
same two steps before they could say anything to the orchestrator - check
``settings.agent_enabled``, then look the reserved ORCHESTRATOR_ID record up out
of the agent store - and each then reached into the run path on its own terms.
This is that path, once. A transport hands it a message and translates what it
raises.

It holds the run service rather than duplicating it: an orchestrator turn is an
agent turn whose agent happens to be the reserved one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..agent import AgentReply, StreamEvent
from ..agent_store import ORCHESTRATOR_ID, AgentRecord, AgentStore
from ..config import Settings
from ..eventbus import EventBus
from ..supervisor import AgentSupervisor
from .errors import AgentDisabled, NoActiveRun, RunAlreadyActive
from .runs import AgentRunService


class OrchestratorTurnService:
    """Start, finish, reset and cancel the orchestrator's turns."""

    def __init__(
        self,
        *,
        settings: Settings,
        agents: AgentStore,
        supervisor: AgentSupervisor,
        runs: AgentRunService,
    ) -> None:
        self._settings = settings
        self._agents = agents
        self._supervisor = supervisor
        self._runs = runs

    async def _record(self) -> AgentRecord:
        """The reserved orchestrator record, read off-loop.

        Off-loop because every store read opens a transaction, which takes
        SQLite's write lock (packages/core/src/scufris_core/engine.py).
        """
        return await asyncio.to_thread(self._agents.get, ORCHESTRATOR_ID)

    async def _enabled_record(self) -> AgentRecord:
        if not self._settings.agent_enabled:
            raise AgentDisabled()
        return await self._record()

    def busy(self) -> bool:
        """Whether the orchestrator has a claimed/queued/running turn - literally
        the condition `AgentRunService.launch` refuses on, so the wake bridge's
        is-busy check and the guard cannot drift apart."""
        return self._runs.active(ORCHESTRATOR_ID)

    async def send(self, message: str) -> AgentReply:
        """One message in, the orchestrator's whole reply out (turn-based).

        Launches the turn and drains its event bus, so it runs through the SAME
        supervised backend path as any agent turn (B5bc) rather than a second
        one that only the non-streaming caller uses. Raises `AgentDisabled`,
        `RunAlreadyActive`, `TurnFailed` or `TurnEndedWithoutReply`.
        """
        orchestrator = await self._enabled_record()
        _run_id, bus = await self._runs.launch(orchestrator, None, message)
        return (await self._runs.drain(bus)).reply

    async def stream(
        self,
        message: str,
        *,
        image_paths: list[str] | None = None,
        on_done: Callable[[], None] | None = None,
    ) -> tuple[str, EventBus[StreamEvent]]:
        """Start a turn and hand back its run id and live event bus.

        The turn runs as a supervised BACKGROUND job, not inside the caller: a
        dropped SSE relay or a Telegram poll that errors does not cancel it.
        ``on_done`` is turn-owned cleanup (the web transport's image tempdir),
        fired when the run ends rather than when a relay disconnects.
        """
        orchestrator = await self._enabled_record()
        return await self._runs.launch(
            orchestrator,
            None,
            message,
            image_paths=image_paths,
            on_done=on_done,
        )

    async def wake(self, prompt: str) -> bool:
        """Grant the orchestrator one turn carrying ``prompt`` (resuming its
        session); True if granted, False if it turned out busy. The wake bridge's
        `launch` contract.

        Called from the run-completion fan-out, which has already released the
        finishing run's serialize key - so this never holds ORCHESTRATOR_ID (no
        self-deadlock, lesson
        `serialize-then-launch-self-deadlocks-on-shared-key`).

        Deliberately NOT gated on ``agent_enabled``: ``auto_wake`` is the switch
        for this path, and gating it here would make a disabled agent raise out
        of a completion callback instead of simply not waking.
        """
        try:
            orchestrator = await self._record()
            await self._runs.launch(orchestrator, None, prompt)
            return True
        except RunAlreadyActive:
            return False

    async def reset(self) -> None:
        """Forget the orchestrator's conversation, so the next turn starts fresh.

        Serialized on ORCHESTRATOR_ID so a reset cannot interleave with an
        in-flight orchestrator turn: its turns run through the supervisor keyed
        on that id, and this takes the same key.
        """
        async with self._supervisor.serialized(ORCHESTRATOR_ID):
            await asyncio.to_thread(self._agents.set_orchestrator_session, None)

    async def cancel(self) -> bool:
        """Stop the orchestrator's in-flight turn; False when it was idle.

        A bool rather than a refusal because the Telegram `/cancel` treats
        "there was nothing to stop" as an answer, not an error.
        """
        try:
            self._runs.cancel(ORCHESTRATOR_ID)
        except NoActiveRun:
            return False
        return True


__all__ = ["OrchestratorTurnService"]
