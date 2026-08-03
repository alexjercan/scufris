"""The transport-independent orchestrator: launching, driving and finishing a turn.

Everything in this package is reachable with no FastAPI app, no Telegram bot and
no lifespan. A transport builds these services, calls them, and translates what
they raise (`errors`) into whatever its own surface speaks - an HTTP status, a
`StreamError` line in a chat.
"""

from __future__ import annotations

from .errors import (
    AgentDisabled,
    AgentProjectMissing,
    NoActiveRun,
    OrchestratorError,
    RunAlreadyActive,
    TurnEndedWithoutReply,
    TurnFailed,
)
from .runs import AgentRunService, TurnStatus
from .turn import OrchestratorTurnService

__all__ = [
    "AgentDisabled",
    "AgentProjectMissing",
    "AgentRunService",
    "NoActiveRun",
    "OrchestratorError",
    "OrchestratorTurnService",
    "RunAlreadyActive",
    "TurnEndedWithoutReply",
    "TurnFailed",
    "TurnStatus",
]
