"""Laying one orchestrator turn out over Telegram messages.

``on_message`` STREAMS a turn as ``StreamEvent`` values (the same events the web
UI renders over SSE), and ``_render_turn`` lays them out message-per-phase:

    a "thinking" message that is edited live as the orchestrator's reasoning
    streams (``StreamReasoningDelta``), one discrete widget message per tool call
    (``StreamTool``), then the final answer as its own message (``StreamDone``).

A "typing..." chat action runs for the whole turn on top of that. When ``stream``
is False only the final answer is sent (the one-message-per-turn behaviour).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress

from ..agent import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTool,
)
from .api import BotApi
from .render import _format_reasoning, _format_tool, markdown_reply, render_reply
from .text import EMPTY_REPLY

logger = logging.getLogger(__name__)

# A Telegram "typing..." status expires after ~5s, so a turn that outlasts it
# needs the action re-sent; keep a little headroom under the 5s window.
_TYPING_INTERVAL = 4.0

# Minimum gap between edits of the live "thinking" message. Telegram rate-limits
# edits to a message (~1/s) and 429s on bursts, so reasoning deltas are coalesced
# and flushed at most this often (the FIRST paint is immediate - lesson
# `dont-gate-streaming-render-on-a-single-raf` - and a phase boundary force-flushes).
DEFAULT_EDIT_INTERVAL = 1.5


async def _keep_typing(api: BotApi, chat_id: int) -> None:
    """Re-send the "typing..." action every ``_TYPING_INTERVAL`` seconds until
    cancelled, so a long turn keeps showing activity."""
    while True:
        await asyncio.sleep(_TYPING_INTERVAL)
        await api.try_typing(chat_id)


async def drive_turn(
    api: BotApi,
    chat_id: int,
    events: AsyncIterator[StreamEvent],
    *,
    stream: bool,
    edit_interval: float,
) -> None:
    """Render one orchestrator turn while the poll loop keeps receiving commands."""
    # Show "typing..." while the turn runs. One action is sent up front so
    # even a fast turn shows activity; a keepalive re-sends it (the status
    # expires after ~5s) until the turn is done. The indicator is best-effort:
    # a failed action must never cost the user their reply (the update's offset
    # has already advanced in poll_once, so aborting here would drop it), so
    # both sends swallow non-cancellation errors.
    await api.try_typing(chat_id)
    typing = asyncio.create_task(_keep_typing(api, chat_id))
    try:
        await _render_turn(
            api, chat_id, events, stream=stream, edit_interval=edit_interval
        )
    finally:
        typing.cancel()
        with suppress(asyncio.CancelledError):
            await typing


async def _render_turn(
    api: BotApi,
    chat_id: int,
    events: AsyncIterator[StreamEvent],
    *,
    stream: bool,
    edit_interval: float,
) -> None:
    """Consume a turn's ``StreamEvent`` stream and render it message-per-phase.

    A live "thinking" message is opened on the first reasoning delta and edited
    (throttled to ``edit_interval``) as more arrive; a ``StreamTool`` CLOSES that
    bubble (so the next reasoning opens a fresh one below, keeping chat order
    chronological) and sends a tool widget; ``StreamDone`` sends the final answer;
    ``StreamError`` sends its friendly ``detail``. When ``stream`` is off, only the
    final answer (``StreamDone``/``StreamError``) is rendered.
    """
    reasoning_id: int | None = None
    reasoning_buf = ""
    last_body = ""
    last_edit = 0.0
    terminal = False

    async def flush_reasoning(force: bool) -> None:
        nonlocal last_body, last_edit
        if reasoning_id is None:
            return
        body = _format_reasoning(reasoning_buf)
        if body == last_body:
            return  # nothing new (Telegram 400s on an unmodified edit)
        now = time.monotonic()
        if not force and now - last_edit < edit_interval:
            return
        ok = await api.edit_message(chat_id, reasoning_id, body, html=True)
        # Always advance the throttle clock (so a failing edit is not retried
        # faster than the interval), but only treat the body as delivered on
        # success - a dropped edit is re-attempted at the next content change.
        last_edit = now
        if ok:
            last_body = body

    async for event in events:
        if isinstance(event, StreamReasoningDelta):
            if not stream:
                continue
            reasoning_buf += event.delta
            if reasoning_id is None:
                # First paint immediate, so activity shows without waiting on
                # the throttle; subsequent deltas coalesce into edits.
                body = _format_reasoning(reasoning_buf)
                reasoning_id = await api.send_message(chat_id, body, html=True)
                last_body = body
                last_edit = time.monotonic()
            else:
                await flush_reasoning(force=False)
        elif isinstance(event, StreamTool):
            if not stream:
                continue
            await flush_reasoning(force=True)
            reasoning_id = None
            reasoning_buf = ""
            last_body = ""
            await api.send_message(chat_id, _format_tool(event.tool), html=True)
        elif isinstance(event, StreamDone):
            await flush_reasoning(force=True)
            reasoning_id = None
            plain = render_reply(event.reply.text, event.reply.tool_calls)
            if plain:
                md = markdown_reply(event.reply.text, event.reply.tool_calls)
                await api.send_reply(chat_id, md, plain)
            else:
                # No text: a fixed plain notice. It is sent WITHOUT a parse
                # mode (its parens are MarkdownV2 specials) rather than
                # converted, matching the empty-reply coalesce contract.
                await api.send_message(chat_id, EMPTY_REPLY)
            terminal = True
        elif isinstance(event, StreamError):
            await flush_reasoning(force=True)
            reasoning_id = None
            await api.send_message(chat_id, event.detail or EMPTY_REPLY)
            terminal = True
        # StreamTextDelta / StreamSessionStarted carry no per-phase message:
        # the answer is rendered once, authoritatively, from StreamDone.
    if not terminal:
        # The stream ended without a done/error frame (unexpected); still say
        # something rather than leave the user with a bare "thinking" message.
        await api.send_message(chat_id, EMPTY_REPLY)
