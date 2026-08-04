"""Tests for the generic run engine: background execution, no request timeout,
concurrency cap, heartbeat + budget guards, and subscriber-independent lifetime.

The event type is a parameter, so these run against a supervisor of plain
strings rather than of agent stream events. That is not a simplification for the
test's sake - it is the property the split is for: nothing in this file could be
written if the engine still knew what an agent turn was. The agent's
instantiation and its two callbacks are covered by `tests/test_supervisor.py`.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from scufris_core import EventBus, RunState, Supervisor

#: The error events this suite's supervisor publishes, and how it recognises one
#: a stream produced itself - the two things `Supervisor` needs to know about
#: `EventT`, in their smallest honest form.
_ERROR_PREFIX = "error:"


def _error_event(detail: str) -> str:
    return f"{_ERROR_PREFIX}{detail}"


def _error_detail(event: str) -> str | None:
    return event[len(_ERROR_PREFIX) :] if event.startswith(_ERROR_PREFIX) else None


def _supervisor(**kwargs: object) -> Supervisor[str]:
    return Supervisor(
        error_event=_error_event,
        error_detail=_error_detail,
        **kwargs,  # type: ignore[arg-type]
    )


def _stream(*events: str) -> Callable[[], AsyncIterator[str]]:
    async def gen() -> AsyncIterator[str]:
        for event in events:
            yield event

    return gen


async def _drain(bus: EventBus[str]) -> list[str]:
    """Read a bus to close (i.e. until its run finishes)."""
    out: list[str] = []
    async for _seq, event in bus.subscribe():
        out.append(event)
    return out


async def test_runs_a_stream_to_completion() -> None:
    sup = _supervisor()
    bus = sup.start("r1", _stream("hi", "done"))
    events = await _drain(bus)

    assert events == ["hi", "done"]
    status = sup.status("r1")
    assert status is not None and status.state == "done"
    assert status.error is None


async def test_a_terminal_error_event_is_recorded_on_run_error() -> None:
    """A stream that ends by yielding a terminal error event and then STOPS
    completes normally, so RunPhase settles DONE - but _drain records the detail
    on run.error so the terminal outcome carries WHY it failed. The owner's
    on_complete reads this off the snapshot."""
    sup = _supervisor()
    bus = sup.start("r-err", _stream("working", _error_event("timed out after 120s")))
    events = await _drain(bus)

    assert events == ["working", "error:timed out after 120s"]
    status = sup.status("r-err")
    assert status is not None
    # The stream finished (a terminal error event is a normal terminal bus
    # event), so the RunPhase is DONE - but the detail is now on run.error.
    assert status.state == "done"
    assert status.error == "timed out after 120s"


async def test_run_survives_subscriber_disconnect() -> None:
    """A relay that abandons the stream does not cancel the run (ADR-001)."""
    released = asyncio.Event()

    async def gen() -> AsyncIterator[str]:
        yield "first"
        # Simulate work continuing after the client goes away.
        await asyncio.sleep(0.02)
        yield "second"
        yield "finished"
        released.set()

    sup = _supervisor()
    bus = sup.start("r-disc", gen)

    # Consume exactly one event, then disconnect (stop iterating).
    async for _seq, _event in bus.subscribe():
        break

    # The run keeps going and reaches a terminal state on its own.
    await asyncio.wait_for(released.wait(), timeout=1.0)
    await asyncio.sleep(0.01)
    status = sup.status("r-disc")
    assert status is not None and status.state == "done"
    # All three events were published even though nobody was listening.
    assert bus.last_seq == 3


async def test_cancel_marks_cancelled_and_closes_stream() -> None:
    """cancel(run_id) stops a live run: the drain's finally aclose()s the source
    stream so ITS cleanup runs (real upstream abort), the run settles terminal
    with cancelled=True on both the live snapshot and the on_complete snapshot,
    and an error event is published so relays end. A second cancel is a no-op."""
    started = asyncio.Event()
    closed = asyncio.Event()

    async def blocking() -> AsyncIterator[str]:
        try:
            yield "partial"
            started.set()
            await asyncio.sleep(3600)  # block until cancelled
            yield "never"  # pragma: no cover
        finally:
            # The stream's own cleanup (here: signal it ran; in a real backend,
            # proc.kill()) fires because _drain aclose()s the generator on cancel.
            closed.set()

    snapshots: list[RunState] = []
    sup = _supervisor()
    bus = sup.start("r-cancel", blocking, on_complete=lambda s: snapshots.append(s))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    assert sup.cancel("r-cancel") is True
    events = await _drain(bus)
    await asyncio.wait_for(closed.wait(), timeout=2.0)

    status = sup.status("r-cancel")
    assert status is not None and status.cancelled is True
    assert snapshots and snapshots[0].cancelled is True
    # The partial made it out, then a terminal error event ended the stream.
    assert events[0] == "partial"
    assert events[-1] == "error:cancelled"
    # Already terminal -> nothing to cancel.
    assert sup.cancel("r-cancel") is False


async def test_cancel_unknown_run_is_false() -> None:
    sup = _supervisor()
    assert sup.cancel("nope") is False


async def test_no_wall_clock_timeout_without_a_budget() -> None:
    """With budget None, a slow run is NOT killed by any default timeout.

    The old model killed a turn at agent_timeout_seconds; the supervisor applies
    no wall-clock cap unless a budget is set, so a turn slower than that former
    limit still completes.
    """

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.15)
        yield "eventually"

    sup = _supervisor()
    bus = sup.start("r-slow", slow, budget_seconds=None, heartbeat_seconds=None)
    events = await _drain(bus)

    assert events == ["eventually"]
    status = sup.status("r-slow")
    assert status is not None and status.state == "done"


async def test_budget_cancels_an_overlong_run() -> None:
    async def forever() -> AsyncIterator[str]:
        while True:
            await asyncio.sleep(0.02)
            yield "tick"

    sup = _supervisor()
    bus = sup.start("r-budget", forever, budget_seconds=0.05)
    events = await _drain(bus)

    status = sup.status("r-budget")
    assert status is not None and status.state == "error"
    assert "budget" in (status.error or "")
    assert events[-1].startswith(_ERROR_PREFIX)


async def test_heartbeat_cancels_a_stalled_run() -> None:
    async def stall() -> AsyncIterator[str]:
        yield "alive"
        await asyncio.sleep(3600)  # no further events -> stalled
        yield "never"

    sup = _supervisor()
    bus = sup.start("r-stall", stall, heartbeat_seconds=0.05)
    events = await _drain(bus)

    status = sup.status("r-stall")
    assert status is not None and status.state == "error"
    assert "within" in (status.error or "")
    assert events[-1].startswith(_ERROR_PREFIX)


async def test_concurrency_cap_queues_extra_runs() -> None:
    """With cap 1, a second run waits until the first frees the slot."""
    gate = asyncio.Event()

    def blocking(tag: str) -> Callable[[], AsyncIterator[str]]:
        async def gen() -> AsyncIterator[str]:
            yield f"{tag}-start"
            await gate.wait()
            yield f"{tag}-done"

        return gen

    sup = _supervisor(max_concurrent=1)
    bus1 = sup.start("c1", blocking("a"), serialize_key="a")
    bus2 = sup.start("c2", blocking("b"), serialize_key="b")
    # Let run 1 grab the only slot and start; run 2 cannot begin.
    await asyncio.sleep(0.02)

    s1, s2 = sup.status("c1"), sup.status("c2")
    assert s1 is not None and s1.state == "running"
    assert s2 is not None and s2.state == "queued"
    assert bus2.last_seq == 0  # run 2 has emitted nothing yet

    gate.set()  # release both
    await _drain(bus1)
    await _drain(bus2)
    assert sup.status("c1").state == "done"  # type: ignore[union-attr]
    assert sup.status("c2").state == "done"  # type: ignore[union-attr]


async def test_same_key_runs_serialize() -> None:
    """Two runs on the same key do not overlap even under a high cap."""
    order: list[str] = []
    first_running = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> AsyncIterator[str]:
        order.append("first-start")
        first_running.set()
        await release_first.wait()
        order.append("first-end")
        yield "1"

    async def second() -> AsyncIterator[str]:
        order.append("second-start")
        yield "2"

    sup = _supervisor(max_concurrent=8)
    bus1 = sup.start("s1", first, serialize_key="chat")
    bus2 = sup.start("s2", second, serialize_key="chat")
    await first_running.wait()
    await asyncio.sleep(0.02)
    # Second is blocked on the shared key while first holds it.
    assert order == ["first-start"]
    release_first.set()
    await _drain(bus1)
    await _drain(bus2)
    assert order == ["first-start", "first-end", "second-start"]


async def test_serialized_waits_for_an_inflight_run() -> None:
    """A session mutation (reset/new/switch) cannot slip in front of a turn that
    was started just before it - the R1.1 ordering fix. The reservation is taken
    synchronously in start(), so even reserving before the turn's task drains,
    the mutation still runs last."""
    order: list[str] = []
    release = asyncio.Event()

    async def turn() -> AsyncIterator[str]:
        order.append("turn-start")
        await release.wait()
        order.append("turn-end")
        yield "x"

    sup = _supervisor()
    bus = sup.start("t", turn, serialize_key="chat")

    async def mutate() -> None:
        async with sup.serialized("chat"):
            order.append("reset")

    # Reserve the mutation right after start(), before the turn task has drained.
    task = asyncio.create_task(mutate())
    await asyncio.sleep(0.02)
    assert "reset" not in order  # blocked behind the in-flight turn
    release.set()
    await _drain(bus)
    await task
    assert order == ["turn-start", "turn-end", "reset"]


async def test_terminal_runs_are_reaped() -> None:
    """Terminal runs are bounded so `_runs` cannot leak (R1.2)."""
    sup = _supervisor(max_history=3)
    for i in range(5):
        bus = sup.start(f"r{i}", _stream(str(i)))
        await _drain(bus)

    assert sup.status("r0") is None  # oldest reaped
    assert sup.status("r1") is None
    assert sup.status("r4") is not None  # newest kept
    assert len(sup.list_runs()) == 3
