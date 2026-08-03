"""The bot inside the app: how it is launched, what it calls, and one real turn.

Three heights. The ``_lifespan`` wiring, which decides whether a bot exists at
all. The orchestrator callbacks ``build_telegram_callbacks`` hands it, driven
against fakes so a busy agent, a disabled one and a backend error are each one
test. And the end-to-end, which boots the REAL app against a mock backend and
drives one receive -> stream -> reply through the production loop, plus the
read-only ``/settings`` and ``/stats`` commands routed the same way.

The bot harness lives in ``tests/test_telegram.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from test_telegram import (
    API,
    BRAIN,
    WRENCH,
    _capture_sends,
    _drain_turns,
    _make_bot,
    _ok,
    _update,
)

import scufris.backends as backends_mod
from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamTool,
)
from scufris.agent_diagnostics import AgentDiagnostics
from scufris.agent_store import ORCHESTRATOR_ID
from scufris.app import create_app
from scufris.backends import Capability
from scufris.config import Settings
from scufris.enums import Backend
from scufris.orchestrator import AgentDisabled, RunAlreadyActive
from scufris.sessions import ToolCall, UsageQuota
from scufris.telegram import (
    CAP_EMPTY,
    CAP_UNSUPPORTED,
    DISABLED_REPLY,
    SETTINGS_USAGE,
    TURN_FAILED_REPLY,
    TelegramBot,
    build_telegram_callbacks,
)

# --- in-process launch (the _lifespan wiring) --------------------------------


class _FakeBot:
    """Stand-in for TelegramBot that records construction and never polls."""

    instances: list[_FakeBot] = []

    def __init__(
        self,
        token: str,
        allowed: Any,
        on_message: Any,
        on_reset: Any,
        on_cancel: Any,
        **kwargs: Any,
    ) -> None:
        self.token = token
        self.allowed = allowed
        self.on_cancel = on_cancel
        self.kwargs = kwargs
        _FakeBot.instances.append(self)

    async def run(self) -> None:

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
        # The stream flag is threaded from settings (default on).
        assert _FakeBot.instances[0].kwargs.get("stream") is True
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
# These drive the REAL streaming on_message/on_reset/on_cancel logic against a
# fake turn service, so each branch the TRANSPORT owns - a refusal mapped to its
# user-facing line, event forwarding, the stop-at-done rule - is revert-sensitive
# without a full app boot. What the SERVICE owns (the agent_enabled guard, the
# ORCHESTRATOR_ID lookup, reset serialization) is proven in
# tests/test_orchestrator_service.py against the real service.


class _FakeBus:
    """A minimal EventBus stand-in: replays a fixed event list to a subscriber."""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def subscribe(
        self, after_seq: int = 0
    ) -> AsyncIterator[tuple[int, StreamEvent]]:
        seq = after_seq
        for event in self._events:
            seq += 1
            yield seq, event


class _FakeTurn:
    """An `OrchestratorTurnService` stand-in: scripted stream, recorded resets."""

    def __init__(
        self,
        events: list[StreamEvent] | None = None,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        self.events = events or []
        self.error = error
        self.cancelled = cancelled
        self.streamed: list[str] = []
        self.resets = 0

    async def stream(self, message: str, **kwargs: Any) -> tuple[str, _FakeBus]:
        self.streamed.append(message)
        if self.error is not None:
            raise self.error
        return "run1", _FakeBus(self.events)

    async def reset(self) -> None:
        self.resets += 1

    async def cancel(self) -> bool:
        return self.cancelled


def _build(turn: _FakeTurn) -> Any:
    """The real factory over a structural test double (cast past the concrete
    service type the production signature declares)."""
    return build_telegram_callbacks(cast(Any, turn))


async def _collect(on_message: Any, text: str) -> list[StreamEvent]:
    return [event async for event in on_message(text)]


async def test_on_message_streams_turn_events() -> None:
    turn = _FakeTurn([StreamDone(reply=AgentReply(text="pong"), session_id="s1")])

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "ping")
    assert len(events) == 1
    assert isinstance(events[0], StreamDone) and events[0].reply.text == "pong"
    assert turn.streamed == ["ping"]


async def test_on_message_forwards_events_until_done() -> None:
    turn = _FakeTurn(
        [
            StreamReasoningDelta(delta="hmm"),
            StreamTool(
                tool=ToolCall(server="scufris", tool="host_stats", status="success")
            ),
            StreamDone(reply=AgentReply(text="ok"), session_id="s1"),
            # Anything after the done frame must not be forwarded.
            StreamReasoningDelta(delta="late"),
        ]
    )

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "hi")
    assert [type(e).__name__ for e in events] == [
        "StreamReasoningDelta",
        "StreamTool",
        "StreamDone",
    ]


async def test_on_message_disabled_agent() -> None:
    turn = _FakeTurn(error=AgentDisabled())

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].detail == DISABLED_REPLY


async def test_on_message_busy_when_a_turn_is_already_active() -> None:
    turn = _FakeTurn(error=RunAlreadyActive())

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError) and "still working" in events[0].detail


async def test_on_message_maps_a_launch_failure_to_the_friendly_line() -> None:
    """Anything that is NOT a busy/disabled refusal still reports a failed turn.

    The narrowing that came with the typed errors is deliberate: a deleted agent
    or a missing project is no longer reported to the operator as "I am busy".
    """
    turn = _FakeTurn(error=RuntimeError("the store blew up"))

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].detail == TURN_FAILED_REPLY


async def test_on_message_maps_backend_error_to_friendly_line() -> None:
    turn = _FakeTurn([StreamError(detail="app-server blew up")])

    on_message, _, _ = _build(turn)

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    # The raw backend detail is not leaked to the chat.
    assert events[0].detail == TURN_FAILED_REPLY


async def test_on_reset_forgets_the_conversation() -> None:
    turn = _FakeTurn()

    _, on_reset, _ = _build(turn)

    await on_reset()

    assert turn.resets == 1
    assert _FakeBot.instances == []


async def test_on_cancel_stops_orchestrator_run() -> None:
    turn = _FakeTurn(cancelled=True)

    _, _, on_cancel = _build(turn)

    assert await on_cancel() is True


async def test_on_cancel_false_when_idle() -> None:
    turn = _FakeTurn(cancelled=False)

    _, _, on_cancel = _build(turn)

    assert await on_cancel() is False


# --- end-to-end: real app + mock backend -------------------------------------
#
# The one test that boots the REAL app and drives a full receive->stream->reply.
# The production `_lifespan` starts `_start_telegram_bot`, whose poll loop pulls a
# respx-stubbed getUpdates, runs a real orchestrator turn on the mock backend, and
# renders the streamed events. A per-test MockBackend.stream override emits a
# reasoning delta + a tool call so the thinking bubble and tool widget are
# exercised through the real turn path.


async def _streaming_stream(
    self: Any,
    settings: Settings,
    prompt: str,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    yield StreamReasoningDelta(delta="deciding which host tool to call")
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
async def test_end_to_end_receive_stream_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The mock backend streams reasoning + a tool + a done frame so the whole
    # phased render is exercised; run() is stubbed so the lifespan wires the real
    # bot without the poll loop.
    monkeypatch.setattr(backends_mod.MockBackend, "stream", _streaming_stream)
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
    app = await asyncio.to_thread(create_app, settings=settings)

    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(42, 100, "hello bot")])
    )
    sent, send_handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=send_handler)
    # A single reasoning delta means the thinking message is only sent (not
    # edited); route editMessageText anyway so a future extra delta cannot fail
    # the test on an unmocked call.
    respx.post(f"{API}/editMessageText").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )
    actions, action_handler = _capture_sends()
    respx.post(f"{API}/sendChatAction").mock(side_effect=action_handler)

    # Run the real lifespan so `_start_telegram_bot` builds the bot with the real
    # `AgentRunService.launch` + EventBus callbacks, then drive one poll_once: a
    # getUpdates batch -> a REAL orchestrator turn (mock backend) -> the streamed
    # render.
    async with app.router.lifespan_context(app):
        bot = app.state.telegram_bot
        assert bot is not None
        await bot.poll_once()
        await _drain_turns(bot)

    assert sent, "the bot never sent anything"
    texts = [m["text"] for m in sent]
    # A thinking bubble (reasoning), a tool widget, and the final answer, in order.
    assert any(BRAIN in t and "deciding" in t for t in texts)
    assert any(WRENCH in t and "host_stats" in t for t in texts)
    # The final answer is the MarkdownV2-rendered body (escaped tool footer).
    assert texts[-1] == "handled: hello bot\n\ntools: host\\_stats"
    # A "typing..." action was shown while the turn ran.
    assert {"chat_id": 100, "action": "typing"} in actions


@respx.mock
async def test_settings_summary_command_renders_overview() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(1, 100, "/settings")])
    )
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []  # not routed to an orchestrator turn
    assert len(sent) == 1
    assert sent[0]["parse_mode"] == "MarkdownV2"
    assert "gpt-5.5" in sent[0]["text"]
    assert "Subcommands" in sent[0]["text"]


@respx.mock
@pytest.mark.parametrize(
    ("sub", "needle"),
    [
        ("health", "codex auth"),
        ("usage", "weekly"),
        ("tools", "host_stats"),
    ],
)
async def test_settings_subcommands_render_detail(sub: str, needle: str) -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(1, 100, f"/settings {sub}")])
    )
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []
    assert len(sent) == 1
    assert needle in sent[0]["text"]


@respx.mock
async def test_settings_unknown_subcommand_replies_usage() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(
        return_value=_ok([_update(1, 100, "/settings bogus")])
    )
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []
    # The usage line is a plain message (no parse mode), not an orchestrator turn.
    assert sent == [{"chat_id": 100, "text": SETTINGS_USAGE}]


@respx.mock
async def test_stats_command_renders_host_snapshot() -> None:
    bot, rec = _make_bot()
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "/stats")]))
    sent, handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=handler)

    await bot.poll_once()
    await _drain_turns(bot)

    assert rec.messages == []
    assert len(sent) == 1
    assert "testbox" in sent[0]["text"]


# --- cross-surface agreement (the diagnostics envelope) -----------------------
#
# These boot the REAL app per backend and drive the real `/settings` commands, so
# what the bot prints is compared against what `AgentDiagnostics` actually returns
# for that backend rather than against a capability table written down here. They
# live in this module, not in the pure-render or transport ones, because the
# service is the thing under test.

_QUOTA_BACKENDS = [Backend.CODEX, Backend.CLAUDE, Backend.OPENCODE, Backend.MOCK]


async def _telegram_settings_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, backend: Backend
) -> tuple[str, str, Capability[UsageQuota]]:
    """The real app's `/settings` summary and `/settings usage` bodies for a
    backend, plus the quota envelope its diagnostics service returns."""
    monkeypatch.setattr(TelegramBot, "run", _noop_run)
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        # An empty codex home, so the ONE backend with a quota reader reports the
        # supported-but-empty state deterministically instead of whatever the
        # developer's real ~/.codex holds.
        codex_home=tmp_path / "codex",
        agent_backend=backend,
        enable_mock_backend=True,
        agent_enabled=True,
        telegram_bot_token="TEST",
        telegram_allowed_chat_ids=[100],
        _env_file=None,  # type: ignore[call-arg]
    )
    app = await asyncio.to_thread(create_app, settings=settings)
    orchestrator = await asyncio.to_thread(app.state.agents.get, ORCHESTRATOR_ID)
    quota = AgentDiagnostics(settings).usage(orchestrator)

    # The opencode health probe reaches for a local opencode server; answer it the
    # way a box without one does, so respx does not fail the test on the call.
    respx.route(host="127.0.0.1").mock(
        side_effect=httpx.ConnectError("no opencode server")
    )
    sent, send_handler = _capture_sends()
    respx.post(f"{API}/sendMessage").mock(side_effect=send_handler)
    bodies: list[str] = []
    async with app.router.lifespan_context(app):
        bot = app.state.telegram_bot
        assert bot is not None
        for offset, command in enumerate(("/settings", "/settings usage")):
            respx.post(f"{API}/getUpdates").mock(
                return_value=_ok([_update(offset + 1, 100, command)])
            )
            await bot.poll_once()
            await _drain_turns(bot)
            bodies.append(sent[-1]["text"])
    return bodies[0], bodies[1], quota


@respx.mock
@pytest.mark.parametrize("backend", _QUOTA_BACKENDS)
async def test_telegram_settings_match_orchestrator_diagnostics(
    backend: Backend, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    summary, usage_body, quota = await _telegram_settings_bodies(
        monkeypatch, tmp_path, backend
    )
    window = quota.value.primary if quota.value else None
    if not quota.supported:
        reading = CAP_UNSUPPORTED.format(backend=backend.value)
    elif window is None:
        reading = CAP_EMPTY
    else:
        reading = f"{window.used_percent:.0f}%"
    assert reading in summary
    assert reading in usage_body
    # The summary's config line is the same record the service answered from.
    assert f"backend: {backend.value}" in summary


@respx.mock
@pytest.mark.parametrize("backend", _QUOTA_BACKENDS)
async def test_telegram_hides_codex_account_data_for_other_backends(
    backend: Backend, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    summary, usage_body, quota = await _telegram_settings_bodies(
        monkeypatch, tmp_path, backend
    )
    if quota.supported:
        pytest.skip(f"the {backend.value} backend reads a quota")
    for body in (summary, usage_body):
        assert CAP_UNSUPPORTED.format(backend=backend.value) in body
        # No percentage, window label, plan or session count leaks from the one
        # backend that HAS an account quota.
        assert "%" not in body
        assert "weekly" not in body
        assert "plan" not in body
        assert "rollout" not in body
