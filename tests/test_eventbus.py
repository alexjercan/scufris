"""Tests for the per-run EventBus: fan-out, replay, and non-blocking publish."""

from __future__ import annotations

import asyncio

from scufris.agent import AgentReply, StreamDone, StreamEvent, StreamTextDelta
from scufris.eventbus import EventBus


async def _collect(bus: EventBus[StreamEvent], after_seq: int = 0) -> list[tuple[int, StreamEvent]]:
    """Drain a subscription to completion (returns when the bus closes)."""
    out: list[tuple[int, StreamEvent]] = []
    async for seq, event in bus.subscribe(after_seq=after_seq):
        out.append((seq, event))
    return out


async def test_fans_out_to_every_subscriber() -> None:
    bus: EventBus[StreamEvent] = EventBus()
    a = asyncio.create_task(_collect(bus))
    b = asyncio.create_task(_collect(bus))
    await asyncio.sleep(0)  # let both register their queues

    bus.publish(StreamTextDelta(delta="one"))
    bus.publish(StreamTextDelta(delta="two"))
    bus.publish(StreamDone(reply=AgentReply(text="done")))
    await asyncio.sleep(0)
    bus.close()

    got_a, got_b = await a, await b
    assert [s for s, _ in got_a] == [1, 2, 3]
    assert [s for s, _ in got_b] == [1, 2, 3]
    assert got_a[0][1].delta == "one"  # type: ignore[union-attr]
    assert got_a[-1][1].reply.text == "done"  # type: ignore[union-attr]


async def test_replays_buffered_events_after_a_seq() -> None:
    """A late subscriber replays only events newer than its Last-Event-ID."""
    bus: EventBus[StreamEvent] = EventBus()
    bus.publish(StreamTextDelta(delta="1"))  # seq 1
    bus.publish(StreamTextDelta(delta="2"))  # seq 2
    bus.publish(StreamTextDelta(delta="3"))  # seq 3

    task = asyncio.create_task(_collect(bus, after_seq=1))
    await asyncio.sleep(0)  # replay 2,3 then park on the live queue
    bus.close()
    got = await task

    assert [s for s, _ in got] == [2, 3]


async def test_replay_then_live_without_duplicates() -> None:
    bus: EventBus[StreamEvent] = EventBus()
    bus.publish(StreamTextDelta(delta="buffered"))  # seq 1
    task = asyncio.create_task(_collect(bus, after_seq=0))
    await asyncio.sleep(0)  # replay seq 1, then park live
    bus.publish(StreamTextDelta(delta="live"))  # seq 2
    await asyncio.sleep(0)
    bus.close()
    got = await task

    assert [s for s, _ in got] == [1, 2]  # seq 1 not delivered twice


async def test_publish_never_blocks_on_a_full_subscriber_queue() -> None:
    """A subscriber that stops draining must not stall the publisher.

    White-box: register a bounded queue and flood the bus; publish is a
    synchronous, non-blocking, drop-oldest fan-out, so every publish returns and
    the lagging queue stays bounded rather than growing without limit.
    """
    bus: EventBus[StreamEvent] = EventBus(subscriber_queue_size=2)
    stuck: "asyncio.Queue[object]" = asyncio.Queue(maxsize=2)
    bus._subscribers.add(stuck)  # noqa: SLF001 - exercising the fan-out directly

    last = 0
    for i in range(100):
        last = bus.publish(StreamTextDelta(delta=str(i)))

    assert last == 100  # all 100 published, none blocked
    assert stuck.qsize() == 2  # bounded despite never being drained


async def test_close_ends_a_live_subscriber() -> None:
    bus: EventBus[StreamEvent] = EventBus()
    task = asyncio.create_task(_collect(bus))
    await asyncio.sleep(0)
    bus.close()
    got = await asyncio.wait_for(task, timeout=1.0)
    assert got == []
    # Publishing after close is a no-op that does not raise.
    assert bus.publish(StreamTextDelta(delta="late")) == 0
