"""The bot itself: the long-poll loop, the chat allowlist, and command dispatch.

Transport policy only. It owns the ``getUpdates`` loop, the chat-id allowlist
(which IS the auth - there is no public webhook), and which handler an update
reaches. It drives the orchestrator through injected callbacks (``on_message`` /
``on_reset`` / ``on_cancel``) rather than any self-HTTP, so it maps the single
allowed chat onto the SAME orchestrator turn path as the landing chat and stays
unit-testable against a respx-stubbed Bot API.

Beyond turns it answers READ-ONLY commands mirroring the web dashboard's
orchestrator settings - ``/settings`` and its ``health`` / ``usage`` / ``tools``
subcommands, and ``/stats``. These are quick reads, not turns: they bypass the
turn/busy machinery and are served by the ``SettingsOps`` providers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from scufris_hostctl import HostActionRecord

from .api import BotApi
from .approvals import ApprovalSurface
from .contracts import (
    ApprovalOps,
    OnCancel,
    OnMessageStream,
    OnReset,
    SettingsOps,
)
from .render import (
    render_health,
    render_settings_summary,
    render_stats,
    render_tools,
    render_usage,
    settings_markdown,
)
from .text import (
    BUSY_REPLY,
    CANCELLED_REPLY,
    DEFAULT_API_BASE,
    HELP_TEXT,
    IDLE_CANCEL_REPLY,
    RESET_REPLY,
    SETTINGS_USAGE,
    _command_arg,
    _command_of,
    _preview,
)
from .turn import DEFAULT_EDIT_INTERVAL, drive_turn

logger = logging.getLogger(__name__)


class TelegramBot:
    """A long-poll Telegram Bot API client bound to one orchestrator.

    Construct it with the bot token, the allowed chat ids, the three
    orchestrator turn callbacks, and the ``settings_ops`` read-only providers;
    then ``await run()`` to poll until cancelled, or ``await poll_once()`` to
    drive a single batch (the seam the tests use).

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
        on_cancel: OnCancel,
        *,
        settings_ops: SettingsOps,
        approval_ops: ApprovalOps | None = None,
        api_base: str = DEFAULT_API_BASE,
        poll_timeout: int = 30,
        stream: bool = True,
        edit_interval: float = DEFAULT_EDIT_INTERVAL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._allowed = frozenset(allowed_chat_ids)
        self._on_message = on_message
        self._on_reset = on_reset
        self._on_cancel = on_cancel
        self._settings = settings_ops
        self._stream = stream
        self._edit_interval = edit_interval
        self._turn_tasks: set[asyncio.Task[None]] = set()
        self._api = BotApi(
            token, api_base=api_base, poll_timeout=poll_timeout, client=client
        )
        self._approvals = ApprovalSurface(self._api, approval_ops, self._allowed)

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
            await self._cancel_turn_tasks()
            await self._api.aclose()
            logger.info("telegram bot stopped")

    async def poll_once(self) -> None:
        """One ``getUpdates`` long-poll batch: fetch, advance the offset, dispatch."""
        updates = await self._api.get_updates()
        if updates:
            logger.info("telegram received %d update(s)", len(updates))
        else:
            logger.debug("telegram received no updates")
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._api.advance(update_id)
            await self._handle_update(update)

    async def announce_proposal(self, record: HostActionRecord) -> None:
        """Tell every allowlisted chat a host action is waiting for a decision."""
        await self._approvals.announce_proposal(record)

    async def announce_decision(self, record: HostActionRecord) -> None:
        """Update what the chats are looking at after a decision, from any surface."""
        await self._approvals.announce_decision(record)

    async def send_digest(self, text: str) -> str:
        """Send a scheduled digest to every allowlisted chat. Returns "" or the error."""
        return await self._approvals.send_digest(text)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        # An inline-keyboard tap is a callback_query, NOT a message: it is a second
        # update shape rather than another command.
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            await self._approvals.handle_callback(callback)
            return
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
        # A reply to a "why?" prompt is a denial reason, not a new orchestrator turn.
        reply_to = message.get("reply_to_message")
        if isinstance(reply_to, dict):
            prompt_id = reply_to.get("message_id")
            if isinstance(prompt_id, int):
                if await self._approvals.handle_reason_reply(chat_id, prompt_id, text):
                    return
        await self._dispatch(chat_id, text)

    async def _dispatch(self, chat_id: int, text: str) -> None:
        command = _command_of(text)
        if command in ("/new", "/reset"):
            logger.info("telegram /new: resetting session for chat %s", chat_id)
            await self._on_reset()
            await self._api.send_message(chat_id, RESET_REPLY)
            return
        if command == "/cancel":
            logger.info("telegram /cancel: cancelling turn for chat %s", chat_id)
            cancelled = await self._on_cancel()
            if cancelled:
                await self._cancel_turn_tasks()
            await self._api.send_message(
                chat_id, CANCELLED_REPLY if cancelled else IDLE_CANCEL_REPLY
            )
            return
        if command in ("/help", "/start"):
            logger.info("telegram %s from chat %s", command, chat_id)
            await self._api.send_message(chat_id, HELP_TEXT)
            return
        if command == "/settings":
            await self._handle_settings(chat_id, _command_arg(text))
            return
        if command == "/approvals":
            await self._approvals.handle_approvals(chat_id)
            return
        if command == "/deny":
            await self._approvals.handle_deny_command(chat_id, text)
            return
        if command == "/stats":
            logger.info("telegram /stats from chat %s", chat_id)
            await self._send_settings(
                chat_id, render_stats(await self._settings.stats())
            )
            return
        logger.info(
            "telegram driving orchestrator turn for chat %s (%d chars)",
            chat_id,
            len(text),
        )
        if self._has_active_turn():
            await self._api.send_message(chat_id, BUSY_REPLY)
            return
        task = asyncio.create_task(self._drive_turn(chat_id, text))
        self._track_turn_task(task)

    async def _handle_settings(self, chat_id: int, sub: str) -> None:
        """Render a `/settings [sub]` read-out: the summary (no/`summary` arg) or a
        health/usage/tools detail; an unknown subcommand replies with the usage
        line rather than an error or an orchestrator turn."""
        logger.info("telegram /settings %s from chat %s", sub or "(summary)", chat_id)
        if sub in ("", "summary"):
            body = render_settings_summary(
                await self._settings.info(),
                await self._settings.health(),
                await self._settings.tools(),
            )
        elif sub == "health":
            body = render_health(await self._settings.health())
        elif sub == "usage":
            body = render_usage(await self._settings.info())
        elif sub == "tools":
            body = render_tools(await self._settings.tools())
        else:
            await self._api.send_message(chat_id, SETTINGS_USAGE)
            return
        await self._send_settings(chat_id, body)

    async def _send_settings(self, chat_id: int, body: str) -> None:
        """Send a rendered `/settings`|`/stats` body as MarkdownV2, falling back to
        the plain body if Telegram rejects it (reuses the turn reply's guarantee)."""
        await self._api.send_reply(chat_id, settings_markdown(body), body)

    async def _drive_turn(self, chat_id: int, text: str) -> None:
        await drive_turn(
            self._api,
            chat_id,
            self._on_message(text),
            stream=self._stream,
            edit_interval=self._edit_interval,
        )

    def _has_active_turn(self) -> bool:
        return any(not task.done() for task in self._turn_tasks)

    def _track_turn_task(self, task: asyncio.Task[None]) -> None:
        self._turn_tasks.add(task)

        def done(done_task: asyncio.Task[None]) -> None:
            self._turn_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.error(
                    "telegram turn task failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(done)

    async def _cancel_turn_tasks(self) -> None:
        tasks = [task for task in self._turn_tasks if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
