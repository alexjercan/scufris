"""Relaying an event bus to a browser as Server-Sent Events.

Shared by the agent turn endpoints and by the host apply/build endpoints, which
carry deliberately different event types - a root command's output must never be
renderable as model text - so the relay is generic over what it forwards and
depends on nothing but the ability to serialize.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, TypeVar

from fastapi import Request
from fastapi.responses import StreamingResponse

from ..eventbus import EventBus


class SseEvent(Protocol):
    """What the SSE relay needs of an event: that it can serialize itself."""

    def model_dump_json(self) -> str: ...


SseEventT = TypeVar("SseEventT", bound=SseEvent)


def last_event_id(request: Request) -> int:
    """The SSE seq a reconnecting client already has, 0 when it is new."""
    raw = request.headers.get("last-event-id")
    return int(raw) if raw and raw.isdigit() else 0


def relay_bus_sse(bus: EventBus[SseEventT], after_seq: int = 0) -> StreamingResponse:
    """Relay an event bus as an SSE response (replay events after ``after_seq``,
    then live). Shared by the agent events + chat endpoints."""

    async def events() -> AsyncIterator[str]:
        yield f":{' ' * 2048}\n\n"
        async for seq, event in bus.subscribe(after_seq=after_seq):
            yield f"id: {seq}\ndata: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )
