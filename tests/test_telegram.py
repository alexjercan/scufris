"""Transport + rendering tests for the Telegram long-poll bot.

respx stubs the Bot API. Most tests drive the transport with a FAKE injected
callback, proving the mechanics (poll / offset / allowlist / commands / typing)
and the T6 live rendering (a "thinking" message edited as reasoning streams, a
widget message per tool call, then the final answer) in isolation. The final
test is the end-to-end: it boots the REAL app + mock backend and drives one
receive->stream->reply through the production `_lifespan` loop.
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
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamTool,
)
from scufris.agent_store import ORCHESTRATOR_ID
from scufris.app import build_telegram_callbacks, create_app
from scufris.config import Settings
from scufris.enums import Backend
from scufris.sessions import ToolCall
from scufris.telegram import (
    EMPTY_REPLY,
    HELP_TEXT,
    RESET_REPLY,
    TelegramBot,
    _command_of,
    _format_reasoning,
    _format_tool,
    render_reply,
)

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

    async def on_message(self, text: str) -> AsyncIterator[StreamEvent]:
        self.messages.append(text)
        yield StreamDone(reply=AgentReply(text=f"reply: {text}"), session_id="s1")

    async def on_reset(self) -> None:
        self.resets += 1


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

    return TelegramBot(
        "TEST",
        (100,),
        on_message,
        on_reset,
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
    # A plain turn (only a StreamDone) renders one final-answer message.
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


# --- live rendering: thinking bubble + tool widgets + answer (T6) ------------


@respx.mock
async def test_streams_reasoning_tool_and_answer() -> None:
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="thinking "),
        StreamReasoningDelta(delta="harder"),
        StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="success")
        ),
        StreamDone(
            reply=AgentReply(
                text="all good",
                tool_calls=[
                    ToolCall(server="scufris", tool="host_stats", status="success")
                ],
            ),
            session_id="s1",
        ),
    ]
    bot = _events_bot(events, edit_interval=0.0)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/editMessageText").mock(side_effect=edit)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # Message-per-phase, chronological: a thinking send, an edit as reasoning
    # accumulates, a tool widget send, then the final answer send.
    assert [kind for kind, _ in calls] == ["send", "edit", "send", "send"]

    thinking = calls[0][1]
    assert thinking["parse_mode"] == "HTML"
    assert BRAIN in thinking["text"] and "thinking" in thinking["text"]

    # The edit carries the accumulated reasoning ("thinking harder").
    assert "harder" in calls[1][1]["text"]

    tool = calls[2][1]
    assert tool["parse_mode"] == "HTML"
    assert (
        WRENCH in tool["text"]
        and "host_stats" in tool["text"]
        and CHECK in tool["text"]
    )

    answer = calls[3][1]
    # The final answer is PLAIN text (no parse_mode) with the T5 tool footer.
    assert "parse_mode" not in answer
    assert answer["text"] == "all good\n\ntools: host_stats"


@respx.mock
async def test_second_tool_opens_a_fresh_thinking_bubble() -> None:
    # reasoning -> tool A -> reasoning -> tool B: each tool closes the current
    # bubble, so the second reasoning opens a NEW thinking message.
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="first"),
        StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="success")
        ),
        StreamReasoningDelta(delta="second"),
        StreamTool(
            tool=ToolCall(server="scufris", tool="list_agents", status="success")
        ),
        StreamDone(reply=AgentReply(text="done"), session_id="s1"),
    ]
    bot = _events_bot(events, edit_interval=0.0)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/editMessageText").mock(side_effect=edit)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # Two thinking sends + two tool sends + one answer send; no edits (each bubble
    # got a single delta before its tool closed it).
    kinds = [kind for kind, _ in calls]
    assert kinds == ["send", "send", "send", "send", "send"]
    sends = [body["text"] for _, body in calls]
    assert BRAIN in sends[0] and "first" in sends[0]
    assert WRENCH in sends[1] and "host_stats" in sends[1]
    assert BRAIN in sends[2] and "second" in sends[2]
    assert WRENCH in sends[3] and "list_agents" in sends[3]
    assert sends[4] == "done"


@respx.mock
async def test_reasoning_edits_are_throttled() -> None:
    # A large edit_interval suppresses every INTERMEDIATE edit; the reasoning tail
    # is still delivered because the done boundary force-flushes past the throttle.
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="one "),
        StreamReasoningDelta(delta="two "),
        StreamReasoningDelta(delta="three"),
        StreamDone(reply=AgentReply(text="ok"), session_id="s1"),
    ]
    bot = _events_bot(events, edit_interval=100.0)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/editMessageText").mock(side_effect=edit)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # First paint (send), the two middle deltas suppressed, ONE forced edit on the
    # done boundary carrying the full reasoning, then the answer send.
    assert [kind for kind, _ in calls] == ["send", "edit", "send"]
    assert "one two three" in calls[1][1]["text"]


@respx.mock
async def test_unchanged_reasoning_is_not_re_edited() -> None:
    # A no-op reasoning delta (empty) does not change the rendered body, so the
    # unchanged-body guard suppresses an edit even with the throttle disabled.
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="abc"),
        StreamReasoningDelta(delta=""),
        StreamDone(reply=AgentReply(text="ok"), session_id="s1"),
    ]
    bot = _events_bot(events, edit_interval=0.0)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/editMessageText").mock(side_effect=edit)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # The thinking send + the answer send, and NO edit (the body never changed).
    assert [kind for kind, _ in calls] == ["send", "send"]


@respx.mock
async def test_post_tool_reasoning_edits_the_new_bubble() -> None:
    # After a tool closes the first bubble, later reasoning must edit the SECOND
    # thinking message (a fresh message_id), proving the bubble state was reset.
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="pre"),
        StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="success")
        ),
        StreamReasoningDelta(delta="post one "),
        StreamReasoningDelta(delta="post two"),
        StreamDone(reply=AgentReply(text="ok"), session_id="s1"),
    ]
    bot = _events_bot(events, edit_interval=0.0)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/editMessageText").mock(side_effect=edit)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # send(bubble#1 msg 1) -> send(tool msg 2) -> send(bubble#2 msg 3) ->
    # edit(msg 3) -> send(answer msg 4).
    assert [kind for kind, _ in calls] == ["send", "send", "send", "edit", "send"]
    edit_body = calls[3][1]
    # The edit targets the SECOND bubble and carries its accumulated reasoning.
    assert edit_body["message_id"] == 3
    assert "post one post two" in edit_body["text"]


@respx.mock
async def test_stream_disabled_sends_only_final_answer() -> None:
    events: list[StreamEvent] = [
        StreamReasoningDelta(delta="unshown"),
        StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="success")
        ),
        StreamDone(
            reply=AgentReply(
                text="all good",
                tool_calls=[
                    ToolCall(server="scufris", tool="host_stats", status="success")
                ],
            ),
            session_id="s1",
        ),
    ]
    bot = _events_bot(events, stream=False)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, _edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    # editMessageText is intentionally NOT routed: with streaming off, no live
    # thinking message is edited (respx would raise if it were called).

    await bot.poll_once()

    assert [kind for kind, _ in calls] == ["send"]
    assert "parse_mode" not in calls[0][1]
    assert calls[0][1]["text"] == "all good\n\ntools: host_stats"


@respx.mock
async def test_stream_error_sends_detail_as_plain_message() -> None:
    bot = _events_bot([StreamError(detail="boom")])
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, _edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    assert calls == [("send", {"chat_id": 100, "text": "boom"})]


@respx.mock
async def test_empty_final_answer_is_coalesced() -> None:
    bot = _events_bot([StreamDone(reply=AgentReply(text=""), session_id="s1")])
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, _edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()

    # Telegram rejects an empty body, so a blank final answer is coalesced.
    assert calls == [("send", {"chat_id": 100, "text": EMPTY_REPLY})]


# --- pure widget formatters --------------------------------------------------


def test_format_reasoning_empty_is_header_only() -> None:
    header = f"{BRAIN} <b>Thinking...</b>"
    assert _format_reasoning("") == header
    assert _format_reasoning("   ") == header


def test_format_reasoning_escapes_and_italicises() -> None:
    out = _format_reasoning("a < b & c")
    assert out.startswith(f"{BRAIN} <b>Thinking...</b>")
    # The reasoning body is HTML-escaped and wrapped in italics.
    assert "<i>a &lt; b &amp; c</i>" in out


def test_format_reasoning_tail_windows_long_text() -> None:
    out = _format_reasoning("head" + ("y" * 5000))
    assert len(out) <= 4096  # fits Telegram's message cap
    assert "..." in out  # trimmed marker
    assert "head" not in out  # the tail is kept, the head dropped


def test_format_reasoning_caps_length_after_escaping() -> None:
    # `<` escapes to `&lt;` (4x), so a raw-length trim would blow past 4096; the
    # escape-then-trim keeps the ESCAPED body bounded.
    out = _format_reasoning("HEAD" + ("<" * 5000))
    assert len(out) <= 4096
    assert "HEAD" not in out  # head trimmed, tail kept
    assert "&lt;" in out  # body is escaped


def test_format_tool_ok_and_failed() -> None:
    ok = _format_tool(ToolCall(server="scufris", tool="host_stats", status="success"))
    assert ok == f"{WRENCH} <b>host_stats</b> {CHECK}"
    failed = _format_tool(
        ToolCall(server="scufris", tool="create_agent", status="error")
    )
    assert failed == f"{WRENCH} <b>create_agent</b> {CROSS}"


def test_format_tool_shows_non_default_server() -> None:
    out = _format_tool(ToolCall(server="the-den", tool="today", status="ok"))
    assert "the-den.today" in out


def test_format_tool_escapes_names() -> None:
    out = _format_tool(ToolCall(server="scufris", tool="a<b", status="success"))
    assert "a&lt;b" in out


# --- render_reply (final-answer tool-summary footer) -------------------------


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
        self,
        token: str,
        allowed: Any,
        on_message: Any,
        on_reset: Any,
        **kwargs: Any,
    ) -> None:
        self.token = token
        self.allowed = allowed
        self.kwargs = kwargs
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
# These drive the REAL streaming on_message/on_reset logic (agent_enabled guard,
# 409->busy, backend-error->friendly line, event forwarding, reset serialization)
# with a fake launch_turn + EventBus, so each branch is revert-sensitive without a
# full app boot.


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


def _settings(tmp_path: Any, **kw: Any) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
        **kw,
    )


def _build(settings: Settings, agents: Any, supervisor: Any, launch: Any) -> Any:
    """Call the real factory with structural test doubles (cast past the concrete
    AgentStore/Supervisor types the production signature declares)."""
    return build_telegram_callbacks(
        settings,
        cast(Any, agents),
        cast(Any, supervisor),
        cast(Any, launch),
    )


async def _collect(on_message: Any, text: str) -> list[StreamEvent]:
    return [event async for event in on_message(text)]


async def test_on_message_streams_turn_events(tmp_path: Any) -> None:
    agents = _FakeAgents()
    captured: list[tuple[Any, Any, str]] = []

    def launch(agent: Any, project: Any, text: str) -> tuple[str, _FakeBus]:
        captured.append((agent, project, text))
        return (
            "run1",
            _FakeBus([StreamDone(reply=AgentReply(text="pong"), session_id="s1")]),
        )

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True), agents, _FakeSupervisor(), launch
    )

    events = await _collect(on_message, "ping")
    assert len(events) == 1
    assert isinstance(events[0], StreamDone) and events[0].reply.text == "pong"
    assert captured == [(f"agent:{ORCHESTRATOR_ID}", None, "ping")]


async def test_on_message_forwards_events_until_done(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, _FakeBus]:
        return (
            "run1",
            _FakeBus(
                [
                    StreamReasoningDelta(delta="hmm"),
                    StreamTool(
                        tool=ToolCall(
                            server="scufris", tool="host_stats", status="success"
                        )
                    ),
                    StreamDone(reply=AgentReply(text="ok"), session_id="s1"),
                    # Anything after the done frame must not be forwarded.
                    StreamReasoningDelta(delta="late"),
                ]
            ),
        )

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
    )

    events = await _collect(on_message, "hi")
    assert [type(e).__name__ for e in events] == [
        "StreamReasoningDelta",
        "StreamTool",
        "StreamDone",
    ]


async def test_on_message_disabled_agent(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, _FakeBus]:
        raise AssertionError("must not launch a turn when the agent is disabled")

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=False),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
    )

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].detail == "The agent is disabled."


async def test_on_message_busy_on_409(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, _FakeBus]:
        raise HTTPException(status_code=409, detail="a run is already active")

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
    )

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError) and "still working" in events[0].detail


async def test_on_message_maps_backend_error_to_friendly_line(tmp_path: Any) -> None:
    def launch(*a: Any) -> tuple[str, _FakeBus]:
        return ("run1", _FakeBus([StreamError(detail="app-server blew up")]))

    on_message, _ = _build(
        _settings(tmp_path, agent_enabled=True),
        _FakeAgents(),
        _FakeSupervisor(),
        launch,
    )

    events = await _collect(on_message, "hi")
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    # The raw backend detail is not leaked to the chat.
    assert events[0].detail == "Sorry - that turn failed. Please try again."


async def test_on_reset_clears_session_serialized(tmp_path: Any) -> None:
    agents = _FakeAgents()
    supervisor = _FakeSupervisor()

    def launch(*a: Any) -> tuple[str, _FakeBus]:  # pragma: no cover - reset path
        raise AssertionError

    _, on_reset = _build(
        _settings(tmp_path, agent_enabled=True), agents, supervisor, launch
    )

    await on_reset()

    assert agents.reset_sessions == [None]
    assert supervisor.serialized_keys == [ORCHESTRATOR_ID]
    assert _FakeBot.instances == []


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
    app = create_app(settings=settings)

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
    # `_launch_agent_turn` + EventBus callbacks, then drive one poll_once: a
    # getUpdates batch -> a REAL orchestrator turn (mock backend) -> the streamed
    # render.
    async with app.router.lifespan_context(app):
        bot = app.state.telegram_bot
        assert bot is not None
        await bot.poll_once()

    assert sent, "the bot never sent anything"
    texts = [m["text"] for m in sent]
    # A thinking bubble (reasoning), a tool widget, and the final answer, in order.
    assert any(BRAIN in t and "deciding" in t for t in texts)
    assert any(WRENCH in t and "host_stats" in t for t in texts)
    assert texts[-1] == "handled: hello bot\n\ntools: host_stats"
    # A "typing..." action was shown while the turn ran.
    assert {"chat_id": 100, "action": "typing"} in actions
