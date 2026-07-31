"""The Bot API wire: the only module that knows Telegram's HTTP shape.

Owns the httpx client, the bot-token base URL and the ``getUpdates`` cursor, and
exposes the five calls the surface makes. Everything above it - dispatch,
approvals, turn rendering - talks to a ``BotApi`` instead of building requests,
which is what lets those three live in separate modules at all.

The send helpers carry the transport's delivery policy, not a rendering one: a
best-effort call (a typing action, a callback answer, an edit) swallows a
failure rather than losing the turn or the decision that already happened, and
``send_reply`` re-sends a rejected MarkdownV2 body as plain text so a reply is
never dropped by formatting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .text import DEFAULT_API_BASE

logger = logging.getLogger(__name__)

# The read timeout must outlast the long poll, which holds the connection open
# for `poll_timeout` seconds; this headroom covers the round trip on top.
_READ_TIMEOUT_HEADROOM = 10.0


def _message_id(resp: httpx.Response) -> int | None:
    """The ``message_id`` from a Bot API send/edit response, or None if absent."""
    try:
        payload = resp.json()
    except ValueError:
        return None
    result = payload.get("result") if isinstance(payload, dict) else None
    mid = result.get("message_id") if isinstance(result, dict) else None
    return mid if isinstance(mid, int) else None


class BotApi:
    """One bot token's Bot API endpoint, plus the long-poll cursor.

    ``offset`` is the next update id to fetch; ``advance`` moves it past an
    update the caller has taken, so a chat whose updates are ignored is not
    re-delivered on every poll. The client is created here unless one is
    injected, and ``aclose`` closes only a client this object owns.
    """

    def __init__(
        self,
        token: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        poll_timeout: int = 30,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.poll_timeout = poll_timeout
        self.offset = 0
        self._base_url = f"{api_base.rstrip('/')}/bot{token}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(poll_timeout + _READ_TIMEOUT_HEADROOM, connect=10.0)
        )

    async def aclose(self) -> None:
        """Close the httpx client, if this object created it."""
        if self._owns_client:
            await self._client.aclose()

    def advance(self, update_id: int) -> None:
        """Move the ``getUpdates`` cursor past ``update_id``."""
        self.offset = max(self.offset, update_id + 1)
        logger.debug("telegram offset advanced to %d", self.offset)

    async def get_updates(self) -> list[dict[str, Any]]:
        """One ``getUpdates`` long-poll batch, as the update dicts it returned."""
        logger.debug(
            "telegram getUpdates (offset=%d, timeout=%ds)",
            self.offset,
            self.poll_timeout,
        )
        resp = await self._client.post(
            f"{self._base_url}/getUpdates",
            json={"offset": self.offset, "timeout": self.poll_timeout},
        )
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("result", []) if isinstance(payload, dict) else []
        return [u for u in result if isinstance(u, dict)]

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        html: bool = False,
        reply_markup: dict[str, Any] | None = None,
    ) -> int | None:
        """Send one message; return its ``message_id`` (so a live message can be
        edited later). ``html`` selects ``parse_mode=HTML`` for the widget messages;
        ``reply_markup`` carries an inline keyboard or a force-reply."""
        logger.debug("telegram sendMessage to chat %s (%d chars)", chat_id, len(text))
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if html:
            payload["parse_mode"] = "HTML"
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        resp = await self._client.post(f"{self._base_url}/sendMessage", json=payload)
        resp.raise_for_status()
        return _message_id(resp)

    async def send_reply(
        self, chat_id: int, markdown_body: str, plain_body: str
    ) -> None:
        """Send a rendered body as MarkdownV2, falling back to plain text.

        The formatted body is posted with ``parse_mode=MarkdownV2``. If Telegram
        rejects it (a 4xx from a missed escape or a malformed entity), the plain
        body is re-sent with NO parse mode, preserving the guarantee that a reply
        is never dropped by formatting. The plain resend is NOT itself guarded (it
        matches ``send_message``): if it too fails the error propagates, exactly as
        a plain send did before markdown rendering."""
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
        await self.send_message(chat_id, plain_body)

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        html: bool = False,
        reply_markup: dict[str, Any] | None = None,
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
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            resp = await self._client.post(
                f"{self._base_url}/editMessageText", json=payload
            )
            resp.raise_for_status()
            return True
        except Exception:
            logger.debug("telegram editMessageText failed; ignoring", exc_info=True)
            return False

    async def answer_callback(self, callback_id: str, text: str) -> None:
        """Answer a callback query so the client stops spinning. Best-effort: a
        failure here must never lose the decision the tap already made."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text[:200]},
            )
            resp.raise_for_status()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("telegram answerCallbackQuery failed; ignoring", exc_info=True)

    async def send_chat_action(self, chat_id: int) -> None:
        resp = await self._client.post(
            f"{self._base_url}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
        )
        resp.raise_for_status()

    async def try_typing(self, chat_id: int) -> None:
        """Send one best-effort "typing..." action. A transient failure is logged,
        not raised: the indicator is cosmetic and must not block the turn."""
        try:
            await self.send_chat_action(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("telegram sendChatAction failed; ignoring", exc_info=True)
