"""Telegram frontend: a thin async httpx long-poll Bot API client.

Transport only. The bot owns the ``getUpdates`` long-poll loop, the chat-id
allowlist (which IS the auth - there is no public webhook), and command
dispatch. It drives the orchestrator through two injected callbacks
(``on_message`` / ``on_reset``) rather than any self-HTTP, so it maps the single
allowed chat onto the SAME orchestrator turn path as the landing chat and stays
unit-testable against a respx-stubbed Bot API.

Reply RENDERING lives here too. ``on_message`` STREAMS a turn as ``StreamEvent``
values (the same events the web UI renders over SSE), and ``_render_turn`` lays
them out message-per-phase (T6, see tasks/20260726-201901/DECISION.md):

    a "thinking" message that is edited live as the orchestrator's reasoning
    streams (``StreamReasoningDelta``), one discrete widget message per tool call
    (``StreamTool``), then the final answer as its own message (``StreamDone``).

A "typing..." chat action runs for the whole turn on top of that. The thinking
and tool messages use emoji + HTML formatting - a deliberate exception to the
repo's ASCII-only convention, scoped to the Telegram rendered surface (the emoji
are ``\\N{...}`` escapes so the SOURCE stays ASCII).

The final answer is rendered from the model's GitHub-flavoured markdown into
Telegram MarkdownV2 (``markdown_reply`` -> ``telegramify_markdown``): headings
become bold, lists become bullets, and a table becomes an aligned monospace code
block, so the user sees formatted output rather than raw ``#``/``|``/``-``. Because
that reintroduces the parse-error risk the old plain-text answer avoided, the send
is guarded two ways: the converter falls back to the raw body on any exception,
and ``_send_reply`` re-sends that plain body with NO ``parse_mode`` if Telegram
rejects the MarkdownV2 message - a reply is never dropped by formatting. When
``stream`` is False the bot falls back to sending only the final answer (the
pre-T6 one-message-per-turn behaviour).
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Any

import httpx
import telegramify_markdown

from .agent import (
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTool,
)
from .sessions import ToolCall

logger = logging.getLogger(__name__)

# Drive one orchestrator turn from a user message, STREAMING its events. The bot
# renders the events (`_render_turn`); the callback owns launching the turn and
# mapping any app-level condition (disabled/busy/failure) to a terminal
# ``StreamError`` whose ``detail`` is the friendly, user-facing line.
OnMessageStream = Callable[[str], AsyncIterator[StreamEvent]]
# Reset the orchestrator session (the `/new` command).
OnReset = Callable[[], Awaitable[None]]

DEFAULT_API_BASE = "https://api.telegram.org"

HELP_TEXT = (
    "Scufris orchestrator bot. Commands:\n"
    "/new (or /reset) - start a fresh conversation (forget context)\n"
    "/help - show this message\n"
    "\n"
    "Any other message is forwarded to the orchestrator."
)

RESET_REPLY = "Started a fresh conversation."

# Shown when a turn produces no final text (Telegram rejects an empty message).
EMPTY_REPLY = "(the orchestrator returned no text)"

# The read timeout must outlast the long poll, which holds the connection open
# for `poll_timeout` seconds; this headroom covers the round trip on top.
_READ_TIMEOUT_HEADROOM = 10.0

# A Telegram "typing..." status expires after ~5s, so a turn that outlasts it
# needs the action re-sent; keep a little headroom under the 5s window.
_TYPING_INTERVAL = 4.0

# Minimum gap between edits of the live "thinking" message. Telegram rate-limits
# edits to a message (~1/s) and 429s on bursts, so reasoning deltas are coalesced
# and flushed at most this often (the FIRST paint is immediate - lesson
# `dont-gate-streaming-render-on-a-single-raf` - and a phase boundary force-flushes).
_DEFAULT_EDIT_INTERVAL = 1.5

# Bound on the ESCAPED reasoning body, kept well under Telegram's 4096-char cap;
# the TAIL is kept (the current train of thought) with a leading ellipsis when
# trimmed. Bounding the escaped length (not the raw) is what makes the cap airtight.
_REASONING_MAX = 3500

# Tool statuses the backends report for a successful call (anything else is
# surfaced as failed). codex/claude report "success"; the mock/other paths may
# use "ok". Shared by the final-answer footer and the live tool widget.
_OK_STATUSES = frozenset({"ok", "success", "completed", "done"})

# Emoji for the widgets, as \N{...} escapes so the SOURCE stays ASCII (the repo's
# ASCII-only convention) while the RENDERED Telegram message shows the glyph.
_EMOJI_THINKING = "\N{BRAIN}"
_EMOJI_TOOL = "\N{WRENCH}"
_EMOJI_OK = "\N{HEAVY CHECK MARK}"
_EMOJI_FAIL = "\N{CROSS MARK}"


def render_reply(text: str, tool_calls: Sequence[ToolCall]) -> str:
    """Render an orchestrator turn's FINAL answer into the single Telegram message.

    The reply text is returned unchanged when the turn made no tool calls. When
    it did, a compact ASCII footer line is appended (blank line + ``tools: ...``)
    listing the unique tool names in call order, with ``xN`` for a repeated tool
    and ``(failed)`` when any call of that tool did not report success. ASCII only
    (no emoji/typographic chars) so the plain-text ``sendMessage`` stays clean and
    the model's own text is never re-interpreted by an HTML ``parse_mode``.

    Empty ``text`` with tool calls yields a footer-only body (still non-empty, so
    the caller's empty-reply coalesce does not swallow a tools-only turn)."""
    if not tool_calls:
        return text
    order: list[str] = []
    counts: dict[str, int] = {}
    failed: set[str] = set()
    for call in tool_calls:
        if call.tool not in counts:
            order.append(call.tool)
            counts[call.tool] = 0
        counts[call.tool] += 1
        if call.status.lower() not in _OK_STATUSES:
            failed.add(call.tool)
    parts: list[str] = []
    for tool in order:
        label = tool
        if counts[tool] > 1:
            label += f" x{counts[tool]}"
        if tool in failed:
            label += " (failed)"
        parts.append(label)
    footer = "tools: " + ", ".join(parts)
    return f"{text}\n\n{footer}" if text else footer


def markdown_reply(text: str, tool_calls: Sequence[ToolCall]) -> str:
    """Render an orchestrator turn's FINAL answer as a Telegram MarkdownV2 body.

    Builds the same combined body as ``render_reply`` (model text + optional
    ASCII ``tools:`` footer) and converts it from GitHub-flavoured markdown to
    Telegram MarkdownV2 via ``telegramify_markdown.markdownify``: a heading
    becomes bold, a list becomes bullets, a GFM table becomes an aligned
    monospace code block (Telegram has no table primitive), and every MarkdownV2
    special char is backslash-escaped. The result MUST be sent with
    ``parse_mode=MarkdownV2`` (see ``_send_reply``).

    Robustness is load-bearing: this REPLACES the pre-markdown "plain text never
    400s" guarantee, so any converter failure falls back to the raw
    ``render_reply`` body (still deliverable as plain text). An empty answer
    yields "" so the caller keeps its empty-reply coalesce (the fixed
    ``EMPTY_REPLY`` notice) instead of formatting an empty string."""
    plain = render_reply(text, tool_calls)
    if not plain:
        return ""
    try:
        return telegramify_markdown.markdownify(plain)
    except Exception:
        logger.warning(
            "telegram markdown conversion failed; sending plain text", exc_info=True
        )
        return plain


def _format_reasoning(buf: str) -> str:
    """The live "thinking" message body for the accumulated reasoning ``buf``.

    HTML (a bold header + the reasoning in italics), so it must be sent with
    ``parse_mode=HTML``. The reasoning is HTML-escaped (it is model text, may hold
    ``<``/``&``) and the ESCAPED body is tail-windowed to ``_REASONING_MAX`` chars -
    the most recent thought is what matters and the whole must fit Telegram's
    4096-char cap. Escaping happens BEFORE the trim on purpose: ``html.escape`` can
    expand one char into up to ~6 (``&amp;``), so trimming the raw text would not
    actually bound the message length. Cutting inside an escaped entity only ever
    drops its leading ``&...`` (the TAIL is kept), so no bare ``&`` survives."""
    text = buf.strip()
    if not text:
        return f"{_EMOJI_THINKING} <b>Thinking...</b>"
    body = html.escape(text)
    if len(body) > _REASONING_MAX:
        body = "..." + body[-_REASONING_MAX:]
    return f"{_EMOJI_THINKING} <b>Thinking...</b>\n\n<i>{body}</i>"


def _format_tool(call: ToolCall) -> str:
    """One tool-call widget message: wrench + tool name + a status check/cross.

    HTML (``parse_mode=HTML``); the tool/server names are HTML-escaped. The server
    is shown only when it is not the default built-in ``scufris`` server, so the
    common case stays terse."""
    ok = call.status.lower() in _OK_STATUSES
    mark = _EMOJI_OK if ok else _EMOJI_FAIL
    name = html.escape(call.tool)
    if call.server and call.server != "scufris":
        name = f"{html.escape(call.server)}.{name}"
    return f"{_EMOJI_TOOL} <b>{name}</b> {mark}"


def _message_id(resp: httpx.Response) -> int | None:
    """The ``message_id`` from a Bot API send/edit response, or None if absent."""
    try:
        payload = resp.json()
    except ValueError:
        return None
    result = payload.get("result") if isinstance(payload, dict) else None
    mid = result.get("message_id") if isinstance(result, dict) else None
    return mid if isinstance(mid, int) else None


class TelegramBot:
    """A long-poll Telegram Bot API client bound to one orchestrator.

    Construct it with the bot token, the allowed chat ids, and the two
    orchestrator callbacks; then ``await run()`` to poll until cancelled, or
    ``await poll_once()`` to drive a single batch (the seam the tests use).

    ``stream`` (default True) renders a turn message-per-phase (thinking + tool
    widgets + answer); False sends only the final answer. ``edit_interval`` bounds
    how often the live thinking message is edited (tests set 0 to force edits).
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: Sequence[int],
        on_message: OnMessageStream,
        on_reset: OnReset,
        *,
        api_base: str = DEFAULT_API_BASE,
        poll_timeout: int = 30,
        stream: bool = True,
        edit_interval: float = _DEFAULT_EDIT_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._allowed = frozenset(allowed_chat_ids)
        self._on_message = on_message
        self._on_reset = on_reset
        self._poll_timeout = poll_timeout
        self._stream = stream
        self._edit_interval = edit_interval
        self._base_url = f"{api_base.rstrip('/')}/bot{token}"
        # getUpdates offset: the next update id to fetch. Advanced past every
        # update we pull (processed or ignored) so a chat we ignore is not
        # re-delivered on every poll.
        self._offset = 0
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(poll_timeout + _READ_TIMEOUT_HEADROOM, connect=10.0)
        )

    async def run(self) -> None:
        """Long-poll ``getUpdates`` until cancelled, dispatching each update.

        A transient poll error (network blip, a 5xx from Telegram) is logged and
        retried after a short back-off rather than killing the loop; only
        cancellation (app shutdown) stops it. The owned httpx client is closed on
        the way out.
        """
        logger.info("telegram bot started (allowlist: %d chat(s))", len(self._allowed))
        try:
            while True:
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("telegram poll failed; backing off")
                    await asyncio.sleep(3.0)
        finally:
            if self._owns_client:
                await self._client.aclose()
            logger.info("telegram bot stopped")

    async def poll_once(self) -> None:
        """One ``getUpdates`` long-poll batch: fetch, advance the offset, dispatch."""
        logger.debug(
            "telegram getUpdates (offset=%d, timeout=%ds)",
            self._offset,
            self._poll_timeout,
        )
        updates = await self._get_updates()
        if updates:
            logger.info("telegram received %d update(s)", len(updates))
        else:
            logger.debug("telegram received no updates")
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = max(self._offset, update_id + 1)
                logger.debug("telegram offset advanced to %d", self._offset)
            await self._handle_update(update)

    async def _get_updates(self) -> list[dict[str, Any]]:
        resp = await self._client.post(
            f"{self._base_url}/getUpdates",
            json={"offset": self._offset, "timeout": self._poll_timeout},
        )
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("result", []) if isinstance(payload, dict) else []
        return [u for u in result if isinstance(u, dict)]

    async def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = message.get("text")
        # Only text messages carry a command/prompt; ignore anything else.
        if not isinstance(chat_id, int) or not isinstance(text, str):
            logger.debug(
                "telegram ignoring non-text update %s", update.get("update_id")
            )
            return
        if chat_id not in self._allowed:
            logger.info("ignoring telegram update from disallowed chat %s", chat_id)
            return
        logger.debug("telegram message from chat %s: %s", chat_id, _preview(text))
        await self._dispatch(chat_id, text)

    async def _dispatch(self, chat_id: int, text: str) -> None:
        command = _command_of(text)
        if command in ("/new", "/reset"):
            logger.info("telegram /new: resetting session for chat %s", chat_id)
            await self._on_reset()
            await self._send_message(chat_id, RESET_REPLY)
            return
        if command in ("/help", "/start"):
            logger.info("telegram %s from chat %s", command, chat_id)
            await self._send_message(chat_id, HELP_TEXT)
            return
        logger.info(
            "telegram driving orchestrator turn for chat %s (%d chars)",
            chat_id,
            len(text),
        )
        # Show "typing..." while the turn runs. One action is sent up front so
        # even a fast turn shows activity; a keepalive re-sends it (the status
        # expires after ~5s) until the turn is done. The indicator is best-effort:
        # a failed action must never cost the user their reply (the update's offset
        # has already advanced in poll_once, so aborting here would drop it), so
        # both sends swallow non-cancellation errors.
        await self._try_typing(chat_id)
        typing = asyncio.create_task(self._keep_typing(chat_id))
        try:
            await self._render_turn(chat_id, self._on_message(text))
        finally:
            typing.cancel()
            with suppress(asyncio.CancelledError):
                await typing

    async def _render_turn(
        self, chat_id: int, events: AsyncIterator[StreamEvent]
    ) -> None:
        """Consume a turn's ``StreamEvent`` stream and render it message-per-phase.

        A live "thinking" message is opened on the first reasoning delta and edited
        (throttled) as more arrive; a ``StreamTool`` CLOSES that bubble (so the next
        reasoning opens a fresh one below, keeping chat order chronological) and
        sends a tool widget; ``StreamDone`` sends the final answer; ``StreamError``
        sends its friendly ``detail``. When ``stream`` is off, only the final answer
        (``StreamDone``/``StreamError``) is rendered.
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
            if not force and now - last_edit < self._edit_interval:
                return
            ok = await self._edit_message(chat_id, reasoning_id, body, html=True)
            # Always advance the throttle clock (so a failing edit is not retried
            # faster than the interval), but only treat the body as delivered on
            # success - a dropped edit is re-attempted at the next content change.
            last_edit = now
            if ok:
                last_body = body

        async for event in events:
            if isinstance(event, StreamReasoningDelta):
                if not self._stream:
                    continue
                reasoning_buf += event.delta
                if reasoning_id is None:
                    # First paint immediate, so activity shows without waiting on
                    # the throttle; subsequent deltas coalesce into edits.
                    body = _format_reasoning(reasoning_buf)
                    reasoning_id = await self._send_message(chat_id, body, html=True)
                    last_body = body
                    last_edit = time.monotonic()
                else:
                    await flush_reasoning(force=False)
            elif isinstance(event, StreamTool):
                if not self._stream:
                    continue
                await flush_reasoning(force=True)
                reasoning_id = None
                reasoning_buf = ""
                last_body = ""
                await self._send_message(chat_id, _format_tool(event.tool), html=True)
            elif isinstance(event, StreamDone):
                await flush_reasoning(force=True)
                reasoning_id = None
                plain = render_reply(event.reply.text, event.reply.tool_calls)
                if plain:
                    md = markdown_reply(event.reply.text, event.reply.tool_calls)
                    await self._send_reply(chat_id, md, plain)
                else:
                    # No text: a fixed plain notice. It is sent WITHOUT a parse
                    # mode (its parens are MarkdownV2 specials) rather than
                    # converted, matching the empty-reply coalesce contract.
                    await self._send_message(chat_id, EMPTY_REPLY)
                terminal = True
            elif isinstance(event, StreamError):
                await flush_reasoning(force=True)
                reasoning_id = None
                await self._send_message(chat_id, event.detail or EMPTY_REPLY)
                terminal = True
            # StreamTextDelta / StreamSessionStarted carry no per-phase message:
            # the answer is rendered once, authoritatively, from StreamDone.
        if not terminal:
            # The stream ended without a done/error frame (unexpected); still say
            # something rather than leave the user with a bare "thinking" message.
            await self._send_message(chat_id, EMPTY_REPLY)

    async def _keep_typing(self, chat_id: int) -> None:
        """Re-send the "typing..." action every ``_TYPING_INTERVAL`` seconds until
        cancelled, so a long turn keeps showing activity."""
        while True:
            await asyncio.sleep(_TYPING_INTERVAL)
            await self._try_typing(chat_id)

    async def _try_typing(self, chat_id: int) -> None:
        """Send one best-effort "typing..." action. A transient failure is logged,
        not raised: the indicator is cosmetic and must not block the turn."""
        try:
            await self._send_chat_action(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("telegram sendChatAction failed; ignoring", exc_info=True)

    async def _send_chat_action(self, chat_id: int) -> None:
        resp = await self._client.post(
            f"{self._base_url}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
        )
        resp.raise_for_status()

    async def _send_message(
        self, chat_id: int, text: str, *, html: bool = False
    ) -> int | None:
        """Send one message; return its ``message_id`` (so a live message can be
        edited later). ``html`` selects ``parse_mode=HTML`` for the widget messages."""
        logger.debug("telegram sendMessage to chat %s (%d chars)", chat_id, len(text))
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if html:
            payload["parse_mode"] = "HTML"
        resp = await self._client.post(f"{self._base_url}/sendMessage", json=payload)
        resp.raise_for_status()
        return _message_id(resp)

    async def _send_reply(
        self, chat_id: int, markdown_body: str, plain_body: str
    ) -> None:
        """Send the final answer as MarkdownV2, falling back to plain text.

        The formatted body is posted with ``parse_mode=MarkdownV2``. If Telegram
        rejects it (a 4xx from a missed escape or a malformed entity), the plain
        ``render_reply`` body is re-sent with NO parse mode, preserving the
        guarantee that a reply is never dropped by formatting. The plain resend
        is NOT itself guarded (it matches ``_send_message``): if it too fails the
        error propagates, exactly as a plain send did before markdown rendering."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": markdown_body,
            "parse_mode": "MarkdownV2",
        }
        try:
            resp = await self._client.post(
                f"{self._base_url}/sendMessage", json=payload
            )
            resp.raise_for_status()
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "telegram MarkdownV2 reply failed; resending as plain text",
                exc_info=True,
            )
        await self._send_message(chat_id, plain_body)

    async def _edit_message(
        self, chat_id: int, message_id: int, text: str, *, html: bool = False
    ) -> bool:
        """Edit a previously-sent message; return whether it succeeded. Best-effort:
        a rate-limit 429 or an "unmodified" 400 is swallowed (the live thinking
        render is cosmetic and must never abort the turn), and the caller uses the
        False return to re-attempt the dropped content at the next change."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if html:
            payload["parse_mode"] = "HTML"
        try:
            resp = await self._client.post(
                f"{self._base_url}/editMessageText", json=payload
            )
            resp.raise_for_status()
            return True
        except Exception:
            logger.debug("telegram editMessageText failed; ignoring", exc_info=True)
            return False


def _preview(text: str, limit: int = 80) -> str:
    """A short, single-line preview of a message for DEBUG logs (never the full
    body, which can be long and may hold sensitive content)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _command_of(text: str) -> str:
    """The leading bot command of a message, lower-cased and de-mentioned.

    Telegram sends group commands as ``/new@mybot``; strip the ``@mention`` so
    the command matches regardless of how the client addressed the bot. A
    message that does not start with a command returns "".
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return ""
    token = stripped.split(maxsplit=1)[0]
    return token.split("@", 1)[0].lower()
