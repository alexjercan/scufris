"""What the orchestrator services refuse, and why.

A service that raised `HTTPException` would only be callable from a FastAPI
route - which is exactly how the turn path came to be untestable without
`create_app`. These are the same refusals, named for the condition rather than
for the status a client happens to see; `scufris/api/errors.py` owns the one
table that turns them into statuses.

Every one of these carries a `detail` that is safe to hand to a client: it says
what the caller did, never what the backend is doing internally.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """A refusal from the turn path, with a client-safe explanation."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class RunAlreadyActive(OrchestratorError):
    """A turn for this agent is already claimed, queued or running.

    The narrow "busy" signal. The callers that hold a turn back until the agent
    is free - the wake bridge, a deferred host decision, the Telegram bot -
    catch THIS and nothing wider, so a deleted or disabled agent is no longer
    silently reported as a race.
    """

    def __init__(self, detail: str = "a run for this agent is already active") -> None:
        super().__init__(detail)


class NoActiveRun(OrchestratorError):
    """The agent has no live run to cancel or to relay events from."""

    def __init__(self, detail: str = "no active run for this agent") -> None:
        super().__init__(detail)


class AgentDisabled(OrchestratorError):
    """The agent is switched off in settings, so no turn may be launched."""

    def __init__(self, detail: str = "agent is disabled") -> None:
        super().__init__(detail)


class AgentProjectMissing(OrchestratorError):
    """The agent is bound to a project that no longer exists.

    Unprocessable rather than not-found: the agent IS there, its binding is the
    thing the operator has to repair.
    """

    def __init__(self, detail: str = "agent's project no longer exists") -> None:
        super().__init__(detail)


class TurnFailed(OrchestratorError):
    """The turn ended on a terminal ``StreamError`` from the backend.

    Carries the backend's own detail: the web surface shows it, and the Telegram
    transport deliberately does not.
    """


class TurnEndedWithoutReply(OrchestratorError):
    """The turn's bus closed with neither a done frame nor an error.

    Nobody's fault the caller can act on, which is why it is the one refusal
    here that means "server bug".
    """

    def __init__(self, detail: str = "turn ended without a reply") -> None:
        super().__init__(detail)


__all__ = [
    "AgentDisabled",
    "AgentProjectMissing",
    "NoActiveRun",
    "OrchestratorError",
    "RunAlreadyActive",
    "TurnEndedWithoutReply",
    "TurnFailed",
]
