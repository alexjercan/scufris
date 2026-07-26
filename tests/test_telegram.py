"""Transport tests for the Telegram long-poll bot.

respx stubs the Bot API. Most tests drive the transport with a FAKE injected
callback, proving the mechanics (poll / offset / allowlist / commands / reply /
typing / rendering) in isolation. The final test is the T5 end-to-end: it boots
the REAL app + mock backend and drives one receive->turn->reply through the
production `_lifespan` loop.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import pytest
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient

import scufris.backends as backends_mod
from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
)
from scufris.agent_store import ORCHESTRATOR_ID
from scufris.app import build_telegram_callbacks, create_app
from scufris.config import Settings
from scufris.enums import Backend
from scufris.sessions import ToolCall
from scufris.telegram import (
    HELP_TEXT,
    RESET_REPLY,
    TelegramBot,
    _command_of,
    render_reply,
)

API = "https://api.telegram.org/botTEST"


def _update(update_id: int, chat_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


def _ok(result: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


class _Recorder:
    """Injected orchestrator callbacks that record what they were driven with."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.resets = 0

    async def on_message(self, text: str) -> str:
        self.messages.append(text)
        return f"reply: {text}"

    async def on_reset(self) -> None:
        self.resets += 1


def _make_bot(
    rec: _Recorder | None = None, allowed: tuple[int, ...] = (100,)
) -> tuple[TelegramBot, _Recorder]:
    rec = rec or _Recorder()
    bot = TelegramBot(
        "TEST",
        allowed,
        rec.on_message,
        rec.on_reset,
        poll_timeout=0,
    )
    return bot, rec


def _capture_sends() -> tuple[list[dict[str, Any]], Any]:
    """A sendMessage handler that records each request body."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    return sent, handler


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

    assert rec.messages == ["hi"]
    assert sent == [{"chat_id": 100, "text": "reply: hi"}]


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
    await bot.poll_once()

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
    assert rec.messages == []
    assert sent == [{"chat_id": 100, "text": RESET_REPLY}]


@respx.mock
async def test_help_command_lists_commands() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/help")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()

    assert rec.messages == []
    assert rec.resets == 0
    assert sent == [{"chat_id": 100, "text": HELP_TEXT}]


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

    assert rec.messages == []
    assert not send_route.called


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/new", "/new"),
        ("/New@scufris_bot arg", "/new"),
        ("  /help  ", "/help"),
        ("hello there", ""),
        ("", ""),
    ],
)
def test_command_of(text: str, expected: str) -> None:
    assert _command_of(text) == expected


# --- typing action (T5 rendering polish) ------------------------------------


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

    # A real turn shows "typing..." at least once (one is sent up front).
    assert actions == [{"chat_id": 100, "action": "typing"}]


@respx.mock
async def test_commands_send_no_typing_action() -> None:
    bot, _ = _make_bot()
    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(1, 100, "/help"), _update(2, 100, "/new")])
    )
    respx.post(f"{API}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    action_route = respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # /help and /new reply instantly - no orchestrator turn, so no typing action.
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

    # The turn still ran and the reply was still sent.
    assert rec.messages == ["hi"]
    assert sent == [{"chat_id": 100, "text": "reply: hi"}]


# --- render_reply (tool-summary footer) -------------------------------------


def _tc(tool: str, status: str = "success", server: str = "scufris") -> ToolCall:
    return ToolCall(server=server, tool=tool, status=status)


def test_render_reply_no_tools_is_unchanged() -> None:
    assert render_reply("hello", []) == "hello"


def test_render_reply_appends_tool_footer() -> None:
    rendered = render_reply("done", [_tc("host_stats"), _tc("list_agents")])
    assert rendered == "done\n\ntools: host_stats, list_agents"


def test_render_reply_counts_repeated_tools_in_call_order() -> None:
    rendered = render_reply(
        "ok",
        [_tc("list_agents"), _tc("host_stats"), _tc("list_agents")],
    )
    # First-seen order, with a count for the repeated tool.
    assert rendered == "ok\n\ntools: list_agents x2, host_stats"


def test_render_reply_marks_failed_calls() -> None:
    rendered = render_reply("oops", [_tc("create_agent", status="error")])
    assert rendered == "oops\n\ntools: create_agent (failed)"


def test_render_reply_empty_text_with_tools_is_footer_only() -> None:
    # A tools-only turn must still yield a non-empty body so the caller's
    # empty-reply coalesce does not swallow it.
    assert render_reply("", [_tc("host_stats")]) == "tools: host_stats"


# --- in-process launch (the _lifespan wiring) --------------------------------


class _FakeBot:
    """Stand-in for TelegramBot that records construction and never polls."""

    instances: list[_FakeBot] = []

    def __init__(
        self, token: str, allowed: Any, on_message: Any, on_reset: Any
    ) -> None:
        self.token = token
        self.allowed = allowed
        _FakeBot.instances.append(self)

    async def run(self) -> None:
        import asyncio

        await asyncio.Event().wait()  # block until cancelled, like the real loop


def test_bot_launches_in_process_when_token_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _FakeBot.instances.clear()
    monkeypatch.setattr("scufris.app.TelegramBot", _FakeBot)
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        telegram_bot_token="TOKEN123",
        telegram_allowed_chat_ids=[100],
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)
    # TestClient's context manager runs the lifespan (startup + shutdown).
    with TestClient(app):
        assert app.state.telegram_task is not None
        assert not app.state.telegram_task.done()
        assert len(_FakeBot.instances) == 1
        assert _FakeBot.instances[0].token == "TOKEN123"
    # After shutdown the lifespan cancels the task.
    assert app.state.telegram_task.cancelled()


def test_no_bot_without_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _FakeBot.instances.clear()
    monkeypatch.setattr("scufris.app.TelegramBot", _FakeBot)
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)
    with TestClient(app):
        assert app.state.telegram_task is None


# --- the orchestrator callbacks (build_telegram_callbacks) -------------------
#
# These drive the REAL on_message/on_reset logic (agent_enabled guard, 409->busy,
# error->failure line, empty->coalesce, reset serialization) with fakes for the
# internal turn path, so each branch is revert-sensitive without a full app boot.


class _FakeAgents:
    def __init__(self) -> None:
        self.reset_sessions: list[str | None] = []

    def get(self, agent_id: str) -> str:
        return f"agent:{agent_id}"

    def set_orchestrator_session(self, session_id: str | None) -> None:
        self.reset_sessions.append(session_id)


class _FakeSupervisor:
    def __init__(self) -> None:
        self.serialized_keys: list[str] = []

    def serialized(self, key: str) -> Any:
        self.serialized_keys.append(key)

        @asynccontextmanager
        async def _cm() -> AsyncIterator[None]:
            yield

        return _cm()


def _settings(tmp_path: Any, **kw: Any) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
        **kw,
    )


def _build(
    settings: Settings, agents: Any, supervisor: Any, launch: Any, drain: Any
) -> Any:
    """Call the real factory with structural test doubles (cast past the concrete
    AgentStore/Supervisor/EventBus types the production signature declares)."""
    return build_telegram_callbacks(
        settings,
        cast(Any, agents),
        cast(Any, supervisor),
        cast(Any, launch),
        cast(Any, drain),
    )


def _done(text: str) -> StreamDone:
    return StreamDone(reply=AgentReply(text=text), session_id="s1")


async def test_on_message_drives_turn_and_returns_reply(tmp_path: Any) -> None:
    agents = _FakeAgents()
    captured: list[tuple[Any, Any, str]] = []

    def launch(agent: Any, project: Any, text: str) -> tuple[str, str]:
        captured.append((agent, project, text))
        return ("run1", "BUS")

    async def drain(bus: str) -> StreamDone:
        assert bus == "BUS"
        return _done("pong")

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        agents,
        _FakeSupervisor(),
        launch,
        drain,
    )

    assert await on_message("ping") == "pong"
    assert captured == [(f"agent:{ORCHESTRATOR_ID}", None, "ping")]


async def test_on_message_disabled_agent(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, str]:
        raise AssertionError("must not launch a turn when the agent is disabled")

    async def drain(bus: Any) -> StreamDone:  # pragma: no cover - never reached
        raise AssertionError

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=False),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
        drain,
    )

    assert await on_message("hi") == "The agent is disabled."


async def test_on_message_busy_on_409(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, str]:
        raise HTTPException(status_code=409, detail="a run is already active")

    async def drain(bus: Any) -> StreamDone:  # pragma: no cover - never reached
        raise AssertionError

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
        drain,
    )

    reply = await on_message("hi")
    assert "still working" in reply


async def test_on_message_reports_turn_error(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, str]:
        return ("run1", "BUS")

    async def drain(bus: Any) -> StreamDone:
        # A backend StreamError surfaces from _drain_turn as a 503.
        raise HTTPException(status_code=503, detail="backend blew up")

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
        drain,
    )

    reply = await on_message("hi")
    assert reply == "Sorry - that turn failed. Please try again."


async def test_on_message_coalesces_empty_reply(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, str]:
        return ("run1", "BUS")

    async def drain(bus: Any) -> StreamDone:
        return _done("")

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
        drain,
    )

    # Telegram rejects an empty message, so a blank reply must be coalesced.
    assert await on_message("hi") == "(the orchestrator returned no text)"


async def test_on_reset_clears_session_serialized(tmp_path: Any) -> None:
    agents = _FakeAgents()
    supervisor = _FakeSupervisor()

    def launch(*a: Any) -> tuple[str, str]:  # pragma: no cover - reset path
        raise AssertionError

    async def drain(bus: Any) -> StreamDone:  # pragma: no cover - reset path
        raise AssertionError

    _, on_reset = _build(
        _settings(tmp_path, agent_enabled=True), agents, supervisor, launch, drain
    )

    await on_reset()

    assert agents.reset_sessions == [None]
    assert supervisor.serialized_keys == [ORCHESTRATOR_ID]
    assert _FakeBot.instances == []


# --- end-to-end: real app + mock backend (T5) -------------------------------
#
# The one test that boots the REAL app and drives a full receive->turn->reply.
# The production `_lifespan` starts `_start_telegram_bot`, whose poll loop pulls a
# respx-stubbed getUpdates, runs a real orchestrator turn on the mock backend, and
# sends the rendered reply. A per-test MockBackend.stream override emits tool calls
# so the tool-summary footer is exercised through the real turn path.


async def _tool_emitting_stream(
    self: Any,
    settings: Settings,
    prompt: str,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    yield StreamTextDelta(delta="on it")
    yield StreamTool(
        tool=ToolCall(server="scufris", tool="host_stats", status="success")
    )
    yield StreamDone(
        reply=AgentReply(
            text=f"handled: {prompt}",
            tool_calls=[
                ToolCall(server="scufris", tool="host_stats", status="success")
            ],
        ),
        session_id=kwargs.get("session_id") or "mock-session",
    )


async def _noop_run(self: Any) -> None:
    """Stub for `TelegramBot.run` so `_start_telegram_bot` builds the REAL bot
    (with the real orchestrator callbacks) without spinning its infinite poll
    loop; the test drives one `poll_once()` deterministically instead."""
    return None


@respx.mock
async def test_end_to_end_receive_turn_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The mock backend calls a tool so the tool-summary footer is exercised, and
    # run() is stubbed so the lifespan wires the real bot without the poll loop.
    monkeypatch.setattr(backends_mod.MockBackend, "stream", _tool_emitting_stream)
    monkeypatch.setattr(TelegramBot, "run", _noop_run)
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        agent_enabled=True,
        telegram_bot_token="TEST",
        telegram_allowed_chat_ids=[100],
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)

    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(42, 100, "hello bot")])
    )
    sent, send_handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=send_handler)
    actions, action_handler = _capture_sends()
    respx.post(f"{API}/sendChatAction").mock(side_effect=action_handler)

    # Run the real lifespan so `_start_telegram_bot` builds the bot with the real
    # `_launch_agent_turn`/`_drain_turn` callbacks, then drive one poll_once: a
    # getUpdates batch -> a REAL orchestrator turn (mock backend) -> the reply.
    async with app.router.lifespan_context(app):
        bot = app.state.telegram_bot
        assert bot is not None
        await bot.poll_once()

    assert sent, "the bot never sent a reply"
    assert sent[0]["chat_id"] == 100
    # The real turn's reply text plus the rendered tool-summary footer.
    assert sent[0]["text"] == "handled: hello bot\n\ntools: host_stats"
    # A "typing..." action was shown while the turn ran.
    assert {"chat_id": 100, "action": "typing"} in actions
