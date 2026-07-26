"""Transport tests for the Telegram long-poll bot.

respx stubs the Bot API; the orchestrator is a FAKE injected callback, so these
prove the transport mechanics (poll / offset / allowlist / commands / reply) in
isolation. The full receive->turn->reply e2e through the real app + mock backend
is T5.
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

from scufris.agent import AgentReply, StreamDone
from scufris.agent_store import ORCHESTRATOR_ID
from scufris.app import build_telegram_callbacks, create_app
from scufris.config import Settings
from scufris.telegram import HELP_TEXT, RESET_REPLY, TelegramBot, _command_of

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
    settings = Settings(web_dist=tmp_path / "absent", state_dir=tmp_path)
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
    return Settings(web_dist=tmp_path / "absent", state_dir=tmp_path, **kw)


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
