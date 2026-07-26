"""Boot the Telegram frontend end to end and watch one message turn into a reply.

This is the acceptance demo for the Telegram bot (spike 20260722-221359, Q5;
tasks T4 transport + T5 rendering). It boots the real FastAPI app in-process
against the in-repo mock backend (no codex, no network, no real Telegram) and
drives the WHOLE receive->turn->reply cycle through the production code path:

    getUpdates (one chat message)  ->  the bot's poll loop
      ->  a real orchestrator turn on the supervised backend
      ->  a "typing..." chat action while the turn runs
      ->  the rendered reply (text + tool-summary footer) via sendMessage.

The Telegram Bot API is stubbed with respx, so the bot's getUpdates / sendMessage
/ sendChatAction calls are served locally and printed instead of hitting
api.telegram.org. The mock backend's stream is overridden to emit a tool call, so
the tool-summary footer (`render_reply`) is exercised for real. The companion
pytest (`test_end_to_end_receive_turn_reply`) asserts the same cycle; this script
is the human-readable walkthrough.

How to run
----------
    python examples/telegram_bot.py

Self-contained: only needs scufris, httpx and respx (all dev deps). Prints each
step and exits 0 when the bot replies, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import respx

import scufris.backends as backends_mod
from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
)
from scufris.app import create_app
from scufris.config import Settings
from scufris.enums import Backend
from scufris.sessions import ToolCall
from scufris.telegram import TelegramBot

API = "https://api.telegram.org/botDEMO"
CHAT_ID = 100
MESSAGE = "how's the box doing?"


async def _demo_stream(
    self: Any,
    settings: Settings,
    prompt: str,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """A mock turn that calls a tool, so the reply carries a tool-summary footer."""
    yield StreamTextDelta(delta="checking...")
    yield StreamTool(
        tool=ToolCall(server="scufris", tool="host_stats", status="success")
    )
    yield StreamDone(
        reply=AgentReply(
            text="The box is healthy: load is low and disk has plenty of room.",
            tool_calls=[
                ToolCall(server="scufris", tool="host_stats", status="success"),
            ],
        ),
        session_id=kwargs.get("session_id") or "mock-session",
    )


def _update(update_id: int, chat_id: int, text: str) -> dict[str, Any]:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


def _ok(result: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


async def _noop_run(self: Any) -> None:
    """Stub for `TelegramBot.run`: let the lifespan build the real bot (with the
    real orchestrator callbacks) without spinning the infinite poll loop; the demo
    drives one deterministic `poll_once()` instead."""
    return None


async def _run(state_dir: Path) -> int:
    # Mock backend that calls a tool, so we demonstrate the tool-summary footer,
    # and a no-op run() so the bot wires the real callbacks without its poll loop.
    backends_mod.MockBackend.stream = _demo_stream  # type: ignore[method-assign]
    TelegramBot.run = _noop_run  # type: ignore[method-assign]

    settings = Settings(
        web_dist=state_dir / "absent",
        state_dir=state_dir,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        agent_enabled=True,
        telegram_bot_token="DEMO",
        telegram_allowed_chat_ids=[CHAT_ID],
    )
    app = create_app(settings=settings)

    sent: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    def capture(bucket: list[dict[str, Any]]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            bucket.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})

        return handler

    print(f"1. A Telegram user (chat {CHAT_ID}) sends: {MESSAGE!r}")
    print("   (delivered to the bot's getUpdates long-poll)")

    with respx.mock:
        respx.post(f"{API}/getUpdates").mock(
            return_value=_ok([_update(1, CHAT_ID, MESSAGE)])
        )
        respx.post(f"{API}/sendMessage").mock(side_effect=capture(sent))
        respx.post(f"{API}/sendChatAction").mock(side_effect=capture(actions))

        # The real _lifespan builds the bot with the real orchestrator callbacks;
        # drive one poll: getUpdates -> a real turn (mock backend) -> the reply.
        async with app.router.lifespan_context(app):
            bot = app.state.telegram_bot
            assert bot is not None
            await bot.poll_once()

    if actions:
        print(f"2. While the turn runs, the bot shows: {actions[0]['action']}...")
    if not sent:
        print("FAIL: the bot never replied.")
        return 1

    print("3. The orchestrator turn (mock backend) finished; the bot replies:")
    print(f"     chat_id: {sent[0]['chat_id']}")
    for line in sent[0]["text"].splitlines():
        print(f"     | {line}")
    print("   (the `tools:` footer is render_reply summarizing the turn's tool calls)")
    print("\nOK: receive -> turn -> reply completed.")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(_run(Path(tmp)))


if __name__ == "__main__":
    sys.exit(main())
