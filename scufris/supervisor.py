"""The agent's supervisor: `scufris_core.Supervisor` filled in for agent turns.

The generic run engine - background execution, the FIFO serialize chain, the
heartbeat and budget guards, `RunState` and `RunPhase` - is
`scufris_core.supervisor`, because a second distribution supervises runs too.
What stays here is the one instantiation the app has had all along: everything
that supervises agent turns takes `AgentSupervisor`, so parameterising the
supervisor did not change a single agent call site's contract.
"""

from __future__ import annotations

import time
from typing import Callable

from scufris_core import Supervisor

from .agent import StreamError, StreamEvent

AgentSupervisor = Supervisor[StreamEvent]


def _agent_error_event(detail: str) -> StreamEvent:
    return StreamError(detail=detail)


def _agent_error_detail(event: StreamEvent) -> str | None:
    """The detail of a terminal failure a backend produced itself.

    A backend that ends a turn in failure yields a ``StreamError`` and then
    STOPS, so the stream completes normally and the except-clauses in
    ``_execute`` never fire. Recognising it here is what lets the terminal
    snapshot carry WHY the turn failed.
    """
    return event.detail if isinstance(event, StreamError) else None


def agent_supervisor(
    *,
    max_concurrent: int = 4,
    max_history: int = 200,
    clock: Callable[[], float] = time.time,
) -> AgentSupervisor:
    """A supervisor for agent turns."""
    return Supervisor(
        error_event=_agent_error_event,
        error_detail=_agent_error_detail,
        max_concurrent=max_concurrent,
        max_history=max_history,
        clock=clock,
    )
