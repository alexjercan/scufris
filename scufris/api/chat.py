"""The orchestrator's own chat: one turn, one streamed turn, and a reset.

Everything about running the turn - the supervised background job, the "chat"
serialization key, the run heartbeat, the session - lives in
`OrchestratorTurnService`. What is here is the HTTP transport over it, plus the
ONE concern that genuinely belongs to this transport and to no other caller: the
base64 image attachment. Neither the Telegram bot nor the wake bridge sends an
image, and a decode failure has to become an SSE frame, which is a shape the
turn service is not allowed to know how to build.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agent import AgentReply
from ..config import Settings
from ..orchestrator import OrchestratorError, OrchestratorTurnService
from .errors import orchestrator_http_error
from .sse import last_event_id, relay_bus_sse


class ImageAttachment(BaseModel):
    """One image attached to a chat turn (base64 payload + its MIME type)."""

    data_base64: str
    mime: str


class ChatRequest(BaseModel):
    message: str
    image: ImageAttachment | None = None


# Reject oversized uploads (decoded) so a bad/huge payload cannot exhaust memory.
_MAX_IMAGE_BYTES = 12 * 1024 * 1024


def write_image_to_temp(image: ImageAttachment) -> tuple[str, str]:
    """Decode a base64 image attachment to a temp file for codex to read.

    Returns ``(tmpdir, path)`` (the caller removes ``tmpdir`` after the turn).
    Raises ``ValueError`` on a non-image type, invalid base64, or oversize payload.
    """
    if not image.mime.startswith("image/"):
        raise ValueError(f"unsupported attachment type: {image.mime}")
    try:
        data = base64.b64decode(image.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("attachment is not valid base64") from exc
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("attachment is too large")
    tmpdir = tempfile.mkdtemp(prefix="scufris-img-")
    ext = mimetypes.guess_extension(image.mime) or ".png"
    path = Path(tmpdir) / f"attachment{ext}"
    path.write_bytes(data)
    return tmpdir, str(path)


@dataclass(frozen=True)
class ChatDeps:
    """What the chat routes read: the turn service, and the one setting that
    decides whether a turn may start at all."""

    settings: Settings
    turn: OrchestratorTurnService


def build_chat_router(deps: ChatDeps) -> APIRouter:
    """The orchestrator's turn-based chat, its SSE stream, and its reset."""
    router = APIRouter()

    @router.post("/api/chat")
    async def post_chat(request: ChatRequest) -> AgentReply:
        """Send one message to the orchestrator and return its reply (turn-based).

        Runs through the SAME supervised backend path as any agent turn (B5bc):
        launch the orchestrator turn, then drain its event bus for the final
        reply. 503 when the agent is disabled, 409 when a turn is already active.
        """
        try:
            return await deps.turn.send(request.message)
        except OrchestratorError as exc:
            raise orchestrator_http_error(exc) from exc

    @router.post("/api/chat/stream")
    async def post_chat_stream(
        request: ChatRequest, http_request: Request
    ) -> StreamingResponse:
        """Send one message and stream live turn progress as SSE.

        The turn runs as a supervised BACKGROUND job (not inside this request):
        this endpoint starts it and relays its event bus. A dropped connection
        therefore does not cancel the turn, and there is no request timeout - the
        turn serializes on the "chat" key and is guarded by the run heartbeat.
        Each `data:` frame is a JSON stream event (`tool`, `text_delta`,
        `reasoning_delta`, `done`, `error`); each carries an SSE `id:` (the bus
        seq) so a reconnect can replay via `Last-Event-ID`.

        Decoding the attachment stays HERE rather than in the turn service: it is
        a base64/MIME concern of THIS transport (neither the bot nor the wake
        bridge sends an image), and its failure mode is an SSE frame the service
        is not allowed to know how to build.
        """
        # Ahead of the decode, not left to `turn.stream`: a disabled agent
        # answers 503 even when the attachment is also bad, rather than a 200
        # carrying an error frame.
        if not deps.settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")

        tmpdir: str | None = None
        image_paths: list[str] | None = None
        image_error: str | None = None
        if request.image is not None:
            try:
                tmpdir, path = write_image_to_temp(request.image)
                image_paths = [path]
            except ValueError as exc:
                image_error = str(exc)

        # A bad image never launches a turn: relay a single error frame instead.
        if image_error is not None:

            async def error_events() -> AsyncIterator[str]:
                yield f":{' ' * 2048}\n\n"
                payload = json.dumps({"kind": "error", "detail": image_error})
                yield f"data: {payload}\n\n"

            return StreamingResponse(
                error_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        # The image tempdir is owned by the turn: cleaned when it finishes, NOT
        # when a relay disconnects.
        def cleanup() -> None:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

        try:
            _run_id, bus = await deps.turn.stream(
                request.message, image_paths=image_paths, on_done=cleanup
            )
        except OrchestratorError as exc:
            raise orchestrator_http_error(exc) from exc

        # Honour a reconnect: replay bus events newer than the client's last seq.
        return relay_bus_sse(bus, last_event_id(http_request))

    @router.post("/api/chat/reset")
    async def post_chat_reset() -> dict[str, bool]:
        """Start a fresh conversation (forget prior context)."""
        if not deps.settings.agent_enabled:
            raise HTTPException(status_code=503, detail="agent is disabled")
        await deps.turn.reset()
        return {"ok": True}

    return router


__all__ = [
    "ChatDeps",
    "ChatRequest",
    "ImageAttachment",
    "build_chat_router",
    "write_image_to_temp",
]
