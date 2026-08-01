"""Live rendering: the thinking bubble, the tool widgets, and the final answer.

One turn becomes several Telegram messages - a "thinking" message edited in
place as reasoning streams, a widget message per tool call, then the answer -
so these assert what the operator's chat looks like WHILE a turn runs, not only
after it. Covers a second tool opening a fresh bubble, edit throttling, an
unchanged bubble not being re-edited, streaming disabled, a stream error, an
empty answer, and the MarkdownV2 send falling back to plain text.

The bot harness lives in ``tests/test_telegram.py``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import respx
from test_telegram import (
    API,
    BRAIN,
    CHECK,
    WRENCH,
    _drain_turns,
    _events_bot,
    _ok,
    _record_calls,
    _update,
)

from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTool,
)
from scufris.sessions import ToolCall
from scufris.telegram import EMPTY_REPLY


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
    await _drain_turns(bot)

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
    # The final answer is MarkdownV2 with the T5 tool footer; the underscore in
    # the tool name is a MarkdownV2 special, so it comes out backslash-escaped.
    assert answer["parse_mode"] == "MarkdownV2"
    assert answer["text"] == "all good\n\ntools: host\\_stats"


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
    await _drain_turns(bot)

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
    await _drain_turns(bot)

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
    await _drain_turns(bot)

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
    await _drain_turns(bot)

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
    await _drain_turns(bot)

    assert [kind for kind, _ in calls] == ["send"]
    # Streaming off still renders the final answer as MarkdownV2 (escaped footer).
    assert calls[0][1]["parse_mode"] == "MarkdownV2"
    assert calls[0][1]["text"] == "all good\n\ntools: host\\_stats"


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
    await _drain_turns(bot)

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
    await _drain_turns(bot)

    # Telegram rejects an empty body, so a blank final answer is coalesced.
    # The fixed notice is sent as PLAIN text (its parens are MarkdownV2 specials).
    assert calls == [("send", {"chat_id": 100, "text": EMPTY_REPLY})]


@respx.mock
async def test_final_answer_is_sent_as_markdownv2() -> None:
    # The final answer path sends the converted body with parse_mode=MarkdownV2.
    events: list[StreamEvent] = [
        StreamDone(reply=AgentReply(text="# Title\n\n- a\n- b"), session_id="s1"),
    ]
    bot = _events_bot(events)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))
    calls, send, _edit = _record_calls()
    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await _drain_turns(bot)

    assert [kind for kind, _ in calls] == ["send"]
    answer = calls[0][1]
    assert answer["parse_mode"] == "MarkdownV2"
    assert "# Title" not in answer["text"]  # heading transformed
    assert "⦁ a" in answer["text"]  # bullet


@respx.mock
async def test_markdownv2_send_failure_falls_back_to_plain() -> None:
    # If Telegram 400s the MarkdownV2 body (a missed escape / bad entity), the
    # bot re-sends the plain render_reply body with NO parse mode - the reply is
    # never dropped by formatting.
    events: list[StreamEvent] = [
        StreamDone(reply=AgentReply(text="risky . answer"), session_id="s1"),
    ]
    bot = _events_bot(events)
    respx.post(f"{API}/getUpdates").mock(return_value=_ok([_update(1, 100, "hi")]))

    calls: list[dict[str, Any]] = []

    def send(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        # Reject only the formatted (MarkdownV2) attempt; accept the plain resend.
        if body.get("parse_mode") == "MarkdownV2":
            return httpx.Response(
                400, json={"ok": False, "description": "can't parse entities"}
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    respx.post(f"{API}/sendMessage").mock(side_effect=send)
    respx.post(f"{API}/sendChatAction").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await bot.poll_once()
    await _drain_turns(bot)

    # First the MarkdownV2 attempt (rejected), then a plain resend (accepted).
    assert len(calls) == 2
    assert calls[0]["parse_mode"] == "MarkdownV2"
    assert "parse_mode" not in calls[1]
    assert calls[1]["text"] == "risky . answer"
