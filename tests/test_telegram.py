"""The Telegram long-poll transport, command dispatch, and the shared bot harness.

respx stubs the Bot API. These drive the transport with a FAKE injected
callback, so the mechanics - poll, offset, the chat allowlist, commands, the
typing action - are proved in isolation from what a turn actually says.

The harness below (``_update``, ``_ok``, ``_Recorder``, ``_make_bot``,
``_events_bot``, ``_capture_sends``, ``_record_calls``, ``_drain_turns``) is
imported by ``tests/test_telegram_stream.py``, ``tests/test_telegram_render.py``
and ``tests/test_telegram_app.py``. It stays here rather than in ``conftest.py``
because it is domain-local: every unrelated test module in the repo would
otherwise carry respx, httpx and a bot in its collection context.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
import respx
from conftest import make_fixture_stats

from scufris.agent import AgentReply, StreamDone, StreamEvent
from scufris.backends import Capability
from scufris.health import AgentHealth, HealthCheck
from scufris.mcp_models import AgentTool
from scufris.sessions import RateWindow, UsageQuota
from scufris.telegram import (
    BUSY_REPLY,
    CANCELLED_REPLY,
    HELP_TEXT,
    IDLE_CANCEL_REPLY,
    RESET_REPLY,
    OrchestratorInfo,
    SettingsOps,
    TelegramBot,
    _command_of,
)
from scufris_host import HostStats

API = "https://api.telegram.org/botTEST"


# The widget glyphs, mirrored from telegram.py as \N{...} escapes so the test
# source stays ASCII like the module under test.
BRAIN = "\N{BRAIN}"


WRENCH = "\N{WRENCH}"


CHECK = "\N{HEAVY CHECK MARK}"


CROSS = "\N{CROSS MARK}"


def _update(update_id: int, chat_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


def _ok(result: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


class _Recorder:
    """Injected orchestrator callbacks that record what they were driven with.

    ``on_message`` STREAMS the turn (T6); the simple recorder yields a single
    ``StreamDone`` so a plain turn renders exactly one final-answer message.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.resets = 0
        self.cancels: list[bool] = []

    async def on_message(self, text: str) -> AsyncIterator[StreamEvent]:
        self.messages.append(text)
        yield StreamDone(reply=AgentReply(text=f"reply: {text}"), session_id="s1")

    async def on_reset(self) -> None:
        self.resets += 1

    async def on_cancel(self) -> bool:
        self.cancels.append(True)
        return True


def _fake_settings_ops() -> SettingsOps:
    """A populated `SettingsOps` for transport/dispatch tests. The render edge
    cases (an unsupported or empty quota, empty tools, degraded health) are
    covered by the pure render-function tests; here the data is fixed so the
    routing is what's under test."""

    async def info() -> OrchestratorInfo:
        return OrchestratorInfo(
            backend="codex",
            model="gpt-5.5",
            auth_mode="chatgpt",
            enabled=True,
            permission_mode="auto",
            quota=Capability.read(
                UsageQuota(
                    plan_type="pro",
                    primary=RateWindow(
                        used_percent=42.0, window_minutes=10080, resets_at=1795000000
                    ),
                    secondary=None,
                )
            ),
        )

    async def health() -> AgentHealth:
        return AgentHealth(
            scufris_version="0.1.0",
            backend="codex",
            backend_version="codex 1.2.3",
            session_count=3,
            last_session=datetime(2026, 7, 20, tzinfo=timezone.utc),
            checks=[
                HealthCheck(
                    name="agent", status="ok", detail="enabled (backend codex)"
                ),
                HealthCheck(
                    name="codex auth",
                    status="warn",
                    detail="unknown",
                    hint="run `codex login`",
                ),
            ],
        )

    async def tools() -> list[AgentTool]:
        return [
            AgentTool(name="host_stats", description="host metrics", server="scufris"),
            AgentTool(
                name="processes",
                description="process list",
                server="scufris",
                enabled=False,
            ),
            AgentTool(
                name="journal_today", description="today's journal", server="den"
            ),
        ]

    async def stats() -> HostStats:
        return make_fixture_stats()

    return SettingsOps(info=info, health=health, tools=tools, stats=stats)


def _make_bot(
    rec: _Recorder | None = None,
    allowed: tuple[int, ...] = (100,),
    *,
    stream: bool = True,
    edit_interval: float = 0.0,
) -> tuple[TelegramBot, _Recorder]:
    rec = rec or _Recorder()
    bot = TelegramBot(
        "TEST",
        allowed,
        rec.on_message,
        rec.on_reset,
        rec.on_cancel,
        settings_ops=_fake_settings_ops(),
        poll_timeout=0,
        stream=stream,
        edit_interval=edit_interval,
    )
    return bot, rec


def _events_bot(
    events: list[StreamEvent], *, stream: bool = True, edit_interval: float = 0.0
) -> TelegramBot:
    """A bot whose on_message replays a fixed ``StreamEvent`` list (render tests)."""

    async def on_message(text: str) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    async def on_reset() -> None:  # pragma: no cover - not exercised here
        return None

    async def on_cancel() -> bool:  # pragma: no cover - not exercised here
        return False

    return TelegramBot(
        "TEST",
        (100,),
        on_message,
        on_reset,
        on_cancel,
        settings_ops=_fake_settings_ops(),
        poll_timeout=0,
        stream=stream,
        edit_interval=edit_interval,
    )


def _capture_sends() -> tuple[list[dict[str, Any]], Any]:
    """A sendMessage handler that records each request body."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    return sent, handler


def _record_calls() -> tuple[list[tuple[str, dict[str, Any]]], Any, Any]:
    """Ordered (kind, body) recorder for sendMessage/editMessageText.

    sendMessage returns an incrementing ``message_id`` so the bot can edit the
    live "thinking" message it just sent."""
    calls: list[tuple[str, dict[str, Any]]] = []
    counter = {"n": 0}

    def send(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        calls.append(("send", json.loads(request.content)))
        return httpx.Response(
            200, json={"ok": True, "result": {"message_id": counter["n"]}}
        )

    def edit(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(("edit", body))
        return httpx.Response(
            200, json={"ok": True, "result": {"message_id": body["message_id"]}}
        )

    return calls, send, edit


async def _drain_turns(bot: TelegramBot) -> None:
    tasks = list(bot._turn_tasks)  # noqa: SLF001 - test waits for owned turn tasks.
    if tasks:
        await asyncio.gather(*tasks)


@respx.mock
async def test_text_message_drives_orchestrator_and_replies() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(5, 100, "hi")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)
    # A real turn also shows a "typing..." action; route it so respx does not
    # reject the unmocked call.
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == ["hi"]
    # A plain turn (only a StreamDone) renders one final-answer message, now sent
    # as MarkdownV2 ("reply: hi" has no specials, so the text is unchanged).
    assert sent == [{"chat_id": 100, "text": "reply: hi", "parse_mode": "MarkdownV2"}]


@respx.mock
async def test_unauthorized_chat_is_ignored() -> None:
    bot, rec = _make_bot(allowed=(100,))
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(7, 999, "hi")]))
    send_route = respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    assert rec.messages == []
    assert not send_route.called


@respx.mock
async def test_offset_advances_past_processed_updates() -> None:
    bot, _ = _make_bot()
    offsets: list[int] = []

    def get_handler(request: httpx.Request) -> httpx.Response:
        offsets.append(json.loads(request.content)["offset"])
        if len(offsets) == 1:
            return _ok([_update(12, 100, "hi")])
        return _ok([])

    respx.post(f"{API}/getUpdates").mock(side_effect=get_handler)
    respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    # The "hi" update drives a real turn, which shows a typing action.
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await _drain_turns(bot)
    await bot.poll_once()
    await _drain_turns(bot)

    # First poll starts at 0; the second asks for last_update_id + 1.
    assert offsets == [0, 13]


@respx.mock
async def test_ignored_chat_still_advances_offset() -> None:
    bot, _ = _make_bot(allowed=(100,))
    offsets: list[int] = []

    def get_handler(request: httpx.Request) -> httpx.Response:
        offsets.append(json.loads(request.content)["offset"])
        if len(offsets) == 1:
            return _ok([_update(20, 999, "hi")])
        return _ok([])

    respx.post(f"{API}/getUpdates").mock(side_effect=get_handler)

    await bot.poll_once()
    await bot.poll_once()

    # An ignored (disallowed) update must not be re-delivered forever.
    assert offsets == [0, 21]


@respx.mock
async def test_new_command_resets_session() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/new")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()

    assert rec.resets == 1
    assert rec.cancels == []
    assert rec.messages == []
    assert sent == [{"chat_id": 100, "text": RESET_REPLY}]


@respx.mock
async def test_cancel_command_cancels_active_turn() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/cancel")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()

    assert rec.cancels == [True]
    assert rec.messages == []
    assert rec.resets == 0
    assert sent == [{"chat_id": 100, "text": CANCELLED_REPLY}]


@respx.mock
async def test_cancel_command_stops_streaming_turn_from_next_poll() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cancels: list[bool] = []

    async def on_message(text: str) -> AsyncIterator[StreamEvent]:
        started.set()
        await release.wait()
        yield StreamDone(reply=AgentReply(text=f"reply: {text}"), session_id="s1")

    async def on_reset() -> None:  # pragma: no cover - not exercised here
        return None

    async def on_cancel() -> bool:
        cancels.append(True)
        return True

    bot = TelegramBot(
        "TEST",
        (100,),
        on_message,
        on_reset,
        on_cancel,
        settings_ops=_fake_settings_ops(),
        poll_timeout=0,
    )
    polls = [
        _ok([_update(1, 100, "slow turn")]),
        _ok([_update(2, 100, "/cancel")]),
    ]
    respx.post(f"{API}/getUpdates").mock(side_effect=polls)
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await asyncio.wait_for(started.wait(), timeout=1)
    await bot.poll_once()

    assert cancels == [True]
    assert sent == [{"chat_id": 100, "text": CANCELLED_REPLY}]
    assert not bot._turn_tasks  # noqa: SLF001 - cancel joins the render task.
    release.set()


@respx.mock
async def test_cancel_command_reports_when_idle() -> None:
    rec = _Recorder()

    async def idle_cancel() -> bool:
        rec.cancels.append(False)
        return False

    bot = TelegramBot(
        "TEST",
        (100,),
        rec.on_message,
        rec.on_reset,
        idle_cancel,
        settings_ops=_fake_settings_ops(),
        poll_timeout=0,
    )
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/cancel")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()

    assert rec.cancels == [False]
    assert rec.messages == []
    assert sent == [{"chat_id": 100, "text": IDLE_CANCEL_REPLY}]


@respx.mock
async def test_text_message_while_turn_active_reports_busy() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    messages: list[str] = []

    async def on_message(text: str) -> AsyncIterator[StreamEvent]:
        messages.append(text)
        started.set()
        await release.wait()
        yield StreamDone(reply=AgentReply(text=f"reply: {text}"), session_id="s1")

    async def on_reset() -> None:  # pragma: no cover - not exercised here
        return None

    async def on_cancel() -> bool:  # pragma: no cover - not exercised here
        return False

    bot = TelegramBot(
        "TEST",
        (100,),
        on_message,
        on_reset,
        on_cancel,
        settings_ops=_fake_settings_ops(),
        poll_timeout=0,
    )
    polls = [
        _ok([_update(1, 100, "first")]),
        _ok([_update(2, 100, "second")]),
    ]
    respx.post(f"{API}/getUpdates").mock(side_effect=polls)
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await asyncio.wait_for(started.wait(), timeout=1)
    await bot.poll_once()
    release.set()
    await _drain_turns(bot)

    assert messages == ["first"]
    assert sent[0] == {"chat_id": 100, "text": BUSY_REPLY}
    assert sent[-1] == {
        "chat_id": 100,
        "text": "reply: first",
        "parse_mode": "MarkdownV2",
    }


@respx.mock
async def test_help_command_lists_commands() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/help")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []
    assert rec.resets == 0
    assert sent == [{"chat_id": 100, "text": HELP_TEXT}]
    assert "/cancel" in HELP_TEXT


@respx.mock
async def test_non_text_update_is_ignored() -> None:
    bot, rec = _make_bot()
    # A photo/sticker update carries no "text"; nothing to dispatch.
    update = {"update_id": 3, "message": {"chat": {"id": 100}}}
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([update]))
    send_route = respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []
    assert not send_route.called


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/new", "/new"),
        ("/New@scufris_bot arg", "/new"),
        ("  /help  ", "/help"),
        ("/cancel@scufris_bot please", "/cancel"),
        ("hello there", ""),
        ("", ""),
    ],
)
def test_command_of(text: str, expected: str) -> None:
    assert _command_of(text) == expected


# --- typing action -----------------------------------------------------------


@respx.mock
async def test_text_turn_shows_typing_action() -> None:
    bot, _ = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(5, 100, "hi")]))
    respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    actions, handler = _capture_sends()
    respx.post(f"{API}/sendChatAction").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    # A real turn shows "typing..." at least once (one is sent up front).
    assert actions == [{"chat_id": 100, "action": "typing"}]


@respx.mock
async def test_commands_send_no_typing_action() -> None:
    bot, _ = _make_bot()
    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok(
            [
                _update(1, 100, "/help"),
                _update(2, 100, "/new"),
                _update(3, 100, "/cancel"),
            ]
        )
    )
    respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    action_route = respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # Commands reply instantly - no orchestrator turn, so no typing action.
    assert not action_route.called


@respx.mock
async def test_typing_action_failure_does_not_block_reply() -> None:
    # The offset advances before dispatch, so a failed (cosmetic) typing action
    # must not abort the turn and drop the user's message.
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(5, 100, "hi")]))
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(500, json={"ok": False})
    )
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    # The turn still ran and the reply was still sent (as MarkdownV2).
    assert rec.messages == ["hi"]
    assert sent == [{"chat_id": 100, "text": "reply: hi", "parse_mode": "MarkdownV2"}]
