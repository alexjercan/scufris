"""Telegram frontend: a thin async httpx long-poll Bot API client.

Transport only. The bot owns the ``getUpdates`` long-poll loop, the chat-id
allowlist (which IS the auth - there is no public webhook), and command
dispatch. It drives the orchestrator through two injected callbacks
(``on_message`` / ``on_reset``) rather than any self-HTTP, so it maps the single
allowed chat onto the SAME orchestrator turn path as the landing chat and stays
unit-testable against a respx-stubbed Bot API.

Reply RENDERING (a "typing..." action while the turn streams, a tool-summary
line) and the end-to-end example live in T5; here a reply is the orchestrator's
final text, sent as one message.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Drive one orchestrator turn from a user message, returning the reply text.
OnMessage = Callable[[str], Awaitable[str]]
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

# The read timeout must outlast the long poll, which holds the connection open
# for `poll_timeout` seconds; this headroom covers the round trip on top.
_READ_TIMEOUT_HEADROOM = 10.0


class TelegramBot:
    """A long-poll Telegram Bot API client bound to one orchestrator.

    Construct it with the bot token, the allowed chat ids, and the two
    orchestrator callbacks; then ``await run()`` to poll until cancelled, or
    ``await poll_once()`` to drive a single batch (the seam the tests use).
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: Sequence[int],
        on_message: OnMessage,
        on_reset: OnReset,
        *,
        api_base: str = DEFAULT_API_BASE,
        poll_timeout: int = 30,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._allowed = frozenset(allowed_chat_ids)
        self._on_message = on_message
        self._on_reset = on_reset
        self._poll_timeout = poll_timeout
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
            await self._send(chat_id, RESET_REPLY)
            return
        if command in ("/help", "/start"):
            logger.info("telegram %s from chat %s", command, chat_id)
            await self._send(chat_id, HELP_TEXT)
            return
        logger.info(
            "telegram driving orchestrator turn for chat %s (%d chars)",
            chat_id,
            len(text),
        )
        reply = await self._on_message(text)
        logger.debug(
            "telegram orchestrator replied to chat %s (%d chars)", chat_id, len(reply)
        )
        await self._send(chat_id, reply)

    async def _send(self, chat_id: int, text: str) -> None:
        logger.debug("telegram sendMessage to chat %s (%d chars)", chat_id, len(text))
        resp = await self._client.post(
            f"{self._base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        resp.raise_for_status()


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
