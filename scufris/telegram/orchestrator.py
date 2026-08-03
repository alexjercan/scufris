"""The bot's orchestrator callbacks, over the shared turn service.

This is the whole of what the Telegram surface does to the orchestrator: send a
message, forget the conversation, stop the current turn. It runs the SAME
supervised turn as the landing chat - one `OrchestratorTurnService`, no
self-HTTP - and its only job on top of that is to turn every refusal into a
terminal ``StreamError`` carrying a line the operator can read, because a
message that silently produces nothing is indistinguishable from a bot that is
down.

Here rather than in `app.py` so the stream/error behavior is testable against a
fake service, with no application factory in the way.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from ..agent import StreamDone, StreamError, StreamEvent
from ..orchestrator import (
    AgentDisabled,
    OrchestratorTurnService,
    RunAlreadyActive,
)
from .contracts import OnCancel, OnMessageStream, OnReset
from .text import BUSY_REPLY, DISABLED_REPLY, TURN_FAILED_REPLY

logger = logging.getLogger(__name__)


def build_telegram_callbacks(
    turn: OrchestratorTurnService,
) -> tuple[OnMessageStream, OnReset, OnCancel]:
    """The bot's three orchestrator callbacks, wired to ``turn``.

    ``on_message`` STREAMS the turn's ``StreamEvent`` values (the bot renders
    them message-per-phase). Every condition the turn can refuse on - the agent
    disabled, a turn already active, a launch failure, or a backend
    ``StreamError`` - is mapped to a terminal ``StreamError`` whose ``detail`` is
    the friendly, user-facing line, so the raw technical detail never reaches the
    chat and a failed turn is always reported, never silently dropped.
    """

    async def on_message(text: str) -> AsyncIterator[StreamEvent]:
        """One orchestrator turn from a chat message -> a stream of StreamEvents."""
        try:
            _run_id, bus = await turn.stream(text)
        except AgentDisabled:
            yield StreamError(detail=DISABLED_REPLY)
            return
        except RunAlreadyActive:
            yield StreamError(detail=BUSY_REPLY)
            return
        except Exception:
            logger.exception("telegram orchestrator turn errored")
            yield StreamError(detail=TURN_FAILED_REPLY)
            return
        try:
            async for _seq, event in bus.subscribe(after_seq=0):
                if isinstance(event, StreamError):
                    # Do not leak a raw backend detail to the chat; log it and
                    # surface the friendly line instead.
                    logger.warning("telegram orchestrator turn error: %s", event.detail)
                    yield StreamError(detail=TURN_FAILED_REPLY)
                    return
                yield event
                if isinstance(event, StreamDone):
                    return
        except Exception:
            logger.exception("telegram orchestrator stream errored")
            yield StreamError(detail=TURN_FAILED_REPLY)

    async def on_reset() -> None:
        """`/new`: forget the orchestrator's conversation, like /api/chat/reset."""
        await turn.reset()

    async def on_cancel() -> bool:
        """`/cancel`: stop the active orchestrator turn, like the web stop button."""
        return await turn.cancel()

    return on_message, on_reset, on_cancel


__all__ = ["build_telegram_callbacks"]
