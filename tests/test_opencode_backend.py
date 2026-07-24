"""Tests for scufris.backends.OpenCodeBackend (opencode serve -> llama.cpp).

stream() is exercised with a fake in-memory OpencodeClient injected via the
``_make_client`` seam; the read path (read_status/read_transcript) uses respx
to stub the synchronous httpx call the backend makes.
"""

from __future__ import annotations

from typing import Any

import httpx
import respx

from scufris.agent import StreamDone, StreamError, StreamTextDelta, StreamTool
from scufris.backends import (
    AgentBackend,
    OpenCodeBackend,
    _opencode_tools_for,
    get_backend,
    parse_opencode_transcript,
)
from scufris.config import Settings
from scufris.opencode_client import (
    Message,
    OpencodeError,
    OpencodeStaleSessionError,
    Session,
)

BASE = "http://opencode.test"


def _settings(**kw: Any) -> Settings:
    return Settings(opencode_url=BASE, **kw)


def _assistant(
    text: str = "hi there", tools: list[str] | None = None
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"type": "step-start"}]
    for t in tools or []:
        parts.append({"type": "tool", "tool": t})
    parts += [{"type": "text", "text": text}, {"type": "step-finish"}]
    return {
        "info": {
            "id": "msg_a",
            "role": "assistant",
            "tokens": {
                "input": 100,
                "output": 6,
                "reasoning": 0,
                "cache": {"read": 40},
            },
            "time": {"created": 1784722453454},
        },
        "parts": parts,
    }


class _FakeClient:
    """In-memory OpencodeClient stand-in for stream() tests."""

    def __init__(
        self,
        reply: dict[str, Any],
        *,
        stale_first: bool = False,
        error: Exception | None = None,
    ) -> None:
        self._reply = reply
        self._stale_first = stale_first
        self._error = error
        self.created: list[str | None] = []
        self.created_metadata: list[dict[str, Any] | None] = []
        self.sent: list[tuple[str, Any]] = []
        self.closed = False

    async def create_session(
        self, *, title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> Session:
        self.created.append(title)
        self.created_metadata.append(metadata)
        return Session(id=f"ses_{len(self.created)}")

    async def send_message(self, session_id: str, request: Any) -> Message:
        self.sent.append((session_id, request))
        if self._error is not None:
            raise self._error
        if self._stale_first and len(self.sent) == 1:
            raise OpencodeStaleSessionError(session_id)
        return Message.model_validate(self._reply)

    async def close(self) -> None:
        self.closed = True


def _backend_with(client: _FakeClient) -> OpenCodeBackend:
    backend = OpenCodeBackend()
    # Inject the fake over the _make_client seam (a real OpencodeClient would open
    # a socket); the fake quacks like one for the methods stream() calls.
    backend._make_client = lambda settings: client  # type: ignore[assignment,method-assign,return-value]
    return backend


async def test_get_backend_resolves_opencode() -> None:
    assert isinstance(get_backend("opencode"), OpenCodeBackend)
    assert get_backend("opencode").name == "opencode"
    assert isinstance(OpenCodeBackend(), AgentBackend)


def _user(text: str) -> dict[str, Any]:
    return {
        "info": {"id": "msg_u", "role": "user", "time": {"created": 1784722453000}},
        "parts": [{"type": "text", "text": text}],
    }


async def test_opencode_backend_delete_session_issues_delete() -> None:
    """delete_session issues DELETE /session/{id} through OpencodeClient.
    200 -> True; a non-200 or network error -> False."""
    with respx.mock(base_url=BASE) as mock:
        route = mock.delete("/session/ses_1").mock(
            return_value=httpx.Response(200, json=True)
        )
        assert await OpenCodeBackend().delete_session(_settings(), "ses_1") is True
    assert route.called
    # A 404 (already gone) -> False, never raises.
    with respx.mock(base_url=BASE) as mock:
        mock.delete("/session/ses_x").mock(return_value=httpx.Response(404))
        assert await OpenCodeBackend().delete_session(_settings(), "ses_x") is False
    # No id -> False without a request.
    assert await OpenCodeBackend().delete_session(_settings(), None) is False


def test_opencode_backend_read_context_maps_status() -> None:
    """read_context maps the status snapshot (window 0; opencode has no window)."""
    with respx.mock(base_url=BASE) as mock:
        mock.get("/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=[_user("hi"), _assistant("yo")])
        )
        ctx = OpenCodeBackend().read_context(_settings(), "ses_1")
    assert ctx is not None
    assert ctx.session_id == "ses_1"
    assert ctx.context_window == 0
    assert ctx.turn_count == 1


def test_permission_mapping_disables_the_right_tools() -> None:
    # manual = read-only: all mutating tools off.
    assert _opencode_tools_for("manual") == {
        "edit": False,
        "write": False,
        "patch": False,
        "bash": False,
    }
    # edit = edits allowed, shell off.
    assert _opencode_tools_for("edit") == {"bash": False}
    # auto = everything on (empty map).
    assert _opencode_tools_for("auto") == {}
    # unknown falls back to the safe manual map.
    assert _opencode_tools_for("bogus") == _opencode_tools_for("manual")


async def test_stream_yields_text_and_done() -> None:
    client = _FakeClient(_assistant("hello from gemma"))
    backend = _backend_with(client)
    events = [
        e async for e in backend.stream(_settings(), "hi", permission_mode="manual")
    ]
    # A fresh session was created (none passed) and the turn sent to it.
    assert client.created == [None]
    sid, request = client.sent[0]
    assert sid == "ses_1"
    assert request.model.modelID == "gemma-4-26B-A4B-it"
    assert request.model.providerID == "llamacpp"
    assert request.tools == {
        "edit": False,
        "write": False,
        "patch": False,
        "bash": False,
    }
    # Events: a text delta then a done carrying the session id + reply.
    assert any(
        isinstance(e, StreamTextDelta) and e.delta == "hello from gemma" for e in events
    )
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "hello from gemma"
    assert done.session_id == "ses_1"
    assert done.reply.usage is not None and done.reply.usage.output_tokens == 6
    assert client.closed is True


async def test_stream_tags_new_session_with_agent_metadata() -> None:
    """A fresh opencode session is created with metadata carrying the owning
    agent id, so ownership is recorded on the provider side (part 2). A resumed
    turn creates nothing, so nothing to tag."""
    client = _FakeClient(_assistant("hi"))
    backend = _backend_with(client)
    _ = [e async for e in backend.stream(_settings(), "hi", agent_id="builder")]
    assert client.created_metadata == [{"agent_id": "builder"}]


async def test_stream_reuses_given_session_and_maps_tools() -> None:
    client = _FakeClient(_assistant("done", tools=["bash", "edit"]))
    backend = _backend_with(client)
    events = [e async for e in backend.stream(_settings(), "go", session_id="ses_x")]
    assert client.created == []  # reused, not created
    assert client.sent[0][0] == "ses_x"
    tool_events = [e for e in events if isinstance(e, StreamTool)]
    assert [t.tool.tool for t in tool_events] == ["bash", "edit"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert len(done.reply.tool_calls) == 2


async def test_stream_recreates_session_on_stale_id() -> None:
    client = _FakeClient(_assistant("recovered"), stale_first=True)
    backend = _backend_with(client)
    events = [e async for e in backend.stream(_settings(), "go", session_id="stale")]
    # First send hit the stale id, second went to a freshly created session.
    assert client.created == [None]
    assert [s for s, _ in client.sent] == ["stale", "ses_1"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.session_id == "ses_1"


async def test_stream_maps_errors_to_stream_error() -> None:
    client = _FakeClient(_assistant(), error=OpencodeError("daemon down"))
    backend = _backend_with(client)
    events = [e async for e in backend.stream(_settings(), "go")]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert "daemon down" in events[0].detail
    assert client.closed is True


def test_read_status_from_messages() -> None:
    payload = [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "q"}]},
        _assistant("the answer", tools=["bash"]),
    ]
    with respx.mock(base_url=BASE) as mock:
        mock.get("/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=payload)
        )
        status = OpenCodeBackend().read_status(_settings(), "ses_1")
    assert status is not None
    assert status.session_id == "ses_1"
    assert status.turns == 1  # one user message
    assert status.tool_calls == 1
    assert status.output_tokens == 6
    assert status.last_message == "the answer"
    assert status.updated_at is not None


def test_read_status_none_paths() -> None:
    assert OpenCodeBackend().read_status(_settings(), None) is None
    with respx.mock(base_url=BASE) as mock:
        mock.get("/session/gone/message").mock(return_value=httpx.Response(404))
        assert OpenCodeBackend().read_status(_settings(), "gone") is None


def test_read_transcript_from_messages() -> None:
    payload = [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "do it"}]},
        _assistant("all done", tools=["bash"]),
    ]
    with respx.mock(base_url=BASE) as mock:
        mock.get("/session/ses_t/message").mock(
            return_value=httpx.Response(200, json=payload)
        )
        msgs = OpenCodeBackend().read_transcript(_settings(), "ses_t")
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].text == "do it"
    assert msgs[1].text == "all done"
    assert msgs[1].tool_calls[0].tool == "bash"
    assert msgs[1].usage is not None and msgs[1].usage.output_tokens == 6
    assert OpenCodeBackend().read_transcript(_settings(), None) == []


def test_parse_transcript_keeps_tool_only_assistant_turn() -> None:
    msgs = parse_opencode_transcript(
        [
            Message.model_validate(
                {"info": {"role": "user"}, "parts": [{"type": "text", "text": "hi"}]}
            ),
            Message.model_validate(
                {
                    "info": {"role": "assistant"},
                    "parts": [{"type": "tool", "tool": "read"}],
                }
            ),
        ]
    )
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].text == ""
    assert msgs[1].tool_calls[0].tool == "read"
