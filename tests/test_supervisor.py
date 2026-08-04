"""Tests for the AGENT's supervisor: the two callbacks `agent_supervisor` fills
in on the generic engine, and nothing else.

The lifecycle itself - background execution, no request timeout, concurrency
cap, heartbeat + budget guards, subscriber-independent lifetime - is
`packages/core/tests/test_supervisor.py`, against a supervisor of plain strings.
What is agent-specific, and only testable here, is that a failure becomes a
`StreamError` on the bus and that a backend's own terminal `StreamError` is
recognised as the run's error detail.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
)
from scufris.supervisor import agent_supervisor
from scufris_core import EventBus


async def _drain(bus: EventBus[StreamEvent]) -> list[StreamEvent]:
    """Read a bus to close (i.e. until its run finishes)."""
    out: list[StreamEvent] = []
    async for _seq, event in bus.subscribe():
        out.append(event)
    return out


async def test_terminal_streamerror_is_recorded_on_run_error() -> None:
    """A backend that ends a turn by yielding a terminal StreamError (idle timeout,
    over-limit line, thread-setup failure) then STOPS completes the stream normally,
    so RunPhase settles DONE - but _drain records the detail on run.error so the
    terminal outcome carries WHY it failed. The persist callback reads this off the
    snapshot to mark the agent ERROR with a diagnostic message."""

    async def stream() -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        yield StreamError(detail="app-server timed out after 120s")

    sup = agent_supervisor()
    bus = sup.start("r-err", stream)
    events = await _drain(bus)

    assert [type(e).__name__ for e in events] == ["StreamTextDelta", "StreamError"]
    status = sup.status("r-err")
    assert status is not None
    # The stream finished (a StreamError is a normal terminal bus event), so the
    # RunPhase is DONE - but the detail is now on run.error for the persist layer.
    assert status.state == "done"
    assert status.error == "app-server timed out after 120s"


async def test_a_failed_run_publishes_a_streamerror() -> None:
    """The other callback: whatever kills a run, relays see a StreamError.

    Without `error_event` an agent relay would end on a bare close and a chat
    surface would have nothing to render, so the terminal event has to be of the
    agent's own event type.
    """

    async def forever() -> AsyncIterator[StreamEvent]:
        while True:
            await asyncio.sleep(0.02)
            yield StreamTextDelta(delta="tick")

    sup = agent_supervisor()
    bus = sup.start("r-budget", forever, budget_seconds=0.05)
    events = await _drain(bus)

    status = sup.status("r-budget")
    assert status is not None and status.state == "error"
    assert "budget" in (status.error or "")
    last = events[-1]
    assert isinstance(last, StreamError)
    assert "budget" in last.detail


async def test_a_completed_agent_turn_settles_done() -> None:
    """The instantiation is wired at all: a real agent stream runs end to end."""

    async def stream() -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="hi")
        yield StreamDone(reply=AgentReply(text="done"))

    sup = agent_supervisor()
    events = await _drain(sup.start("r1", stream))

    assert [type(e).__name__ for e in events] == ["StreamTextDelta", "StreamDone"]
    status = sup.status("r1")
    assert status is not None and status.state == "done" and status.error is None
