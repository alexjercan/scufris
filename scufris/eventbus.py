"""A per-run event bus: one publisher (the agent worker), many subscribers (SSE
relays), with a bounded replay buffer.

This is the ADR-001 seam (tasks/20260720-221748/SPIKE.md): the agent run is
decoupled from any HTTP request. The worker publishes normalized ``StreamEvent``
values here; each open SSE relay ``subscribe``s and forwards them. Because a
subscriber is just a reader, it can drop and reconnect (replaying from ``seq``
via the ``Last-Event-ID`` header) without touching the run, and a slow or dead
subscriber never blocks the publisher or the other subscribers.

The buffer is in-memory and bounded; restart-durable replay from the codex
rollout / claude session log is future work (A2+).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, cast

from .agent import StreamEvent

# Sentinel pushed to every subscriber queue on close() so its generator ends.
_CLOSE = object()


def _offer(queue: "asyncio.Queue[object]", item: object) -> None:
    """Enqueue ``item`` without ever blocking the caller (the publisher).

    On a full queue the oldest item is evicted to make room, so a subscriber
    that has stopped draining loses old events rather than stalling the bus.
    With ``maxsize >= 1`` the evict-then-put always succeeds, so a ``_CLOSE``
    sentinel is guaranteed to be delivered.
    """
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass


class EventBus:
    """A single run's event stream: monotonic ``seq``, bounded replay, fan-out.

    Not thread-safe; all access is expected on one asyncio event loop (no
    ``await`` happens between reading and mutating shared state in ``publish`` /
    subscribe setup, so operations are atomic with respect to each other).
    """

    def __init__(self, *, buffer_size: int = 256, subscriber_queue_size: int = 1024):
        self._buffer: deque[tuple[int, StreamEvent]] = deque(maxlen=buffer_size)
        self._subscribers: set["asyncio.Queue[object]"] = set()
        self._seq = 0
        self._sub_maxsize = subscriber_queue_size
        self._closed = False

    @property
    def last_seq(self) -> int:
        """The seq of the most recently published event (0 before any)."""
        return self._seq

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(self, event: StreamEvent) -> int:
        """Append ``event`` to the buffer and fan it out. Returns its seq.

        A no-op returning the current seq once the bus is closed.
        """
        if self._closed:
            return self._seq
        self._seq += 1
        seq = self._seq
        self._buffer.append((seq, event))
        for queue in self._subscribers:
            _offer(queue, (seq, event))
        return seq

    def close(self) -> None:
        """End every current and future subscriber; publishing becomes a no-op."""
        if self._closed:
            return
        self._closed = True
        for queue in self._subscribers:
            _offer(queue, _CLOSE)

    async def subscribe(
        self, after_seq: int = 0
    ) -> AsyncIterator[tuple[int, StreamEvent]]:
        """Yield ``(seq, event)`` pairs: buffered replay then live.

        Replays buffered events with ``seq > after_seq`` (the ``Last-Event-ID``
        reconnect hook), then streams live events. If the bus is already closed,
        only the replay is yielded. Duplicate seqs that appear both in the replay
        snapshot and the live queue are skipped.
        """
        # Register + snapshot atomically (no await between) so no event slips
        # through the gap between snapshotting the buffer and joining the fan-out.
        queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=self._sub_maxsize)
        self._subscribers.add(queue)
        replay = [(seq, event) for (seq, event) in self._buffer if seq > after_seq]
        already_closed = self._closed
        try:
            last = after_seq
            for seq, event in replay:
                last = seq
                yield seq, event
            if already_closed:
                return
            while True:
                item = await queue.get()
                if item is _CLOSE:
                    return
                seq, event = cast("tuple[int, StreamEvent]", item)
                if seq <= last:
                    continue  # already delivered in the replay
                last = seq
                yield seq, event
        finally:
            self._subscribers.discard(queue)
