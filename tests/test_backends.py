"""Tests for the AgentBackend interface: CodexBackend (stream delegation +
rollout status), MockBackend, and the get_backend factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
)
from scufris.backends import (
    AgentBackend,
    BackendStatus,
    ClaudeBackend,
    CodexBackend,
    MockBackend,
    get_backend,
    parse_claude_stream,
    parse_claude_transcript,
)
from scufris.config import Settings


def _write_rollout(
    home: Path,
    session_id: str,
    *,
    cwd: str,
    turns: int = 1,
    tools: int = 0,
    window: int = 258400,
    last_answer: str = "hi",
) -> Path:
    """A minimal rollout (same shape as test_sessions), for read_status."""
    day = home / "sessions" / "2026" / "07" / "19"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-07-19T14-39-30-{session_id}.jsonl"
    events: list[dict[str, Any]] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": "2026-07-19T14:39:30.556Z",
                "cwd": cwd,
                "originator": "codex_exec",
                "git": {"branch": "main"},
            },
        }
    ]
    for i in range(turns):
        events.append(
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": f"turn {i}"},
            }
        )
        events.append(
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": last_answer},
            }
        )
    for _ in range(tools):
        events.append({"type": "event_msg", "payload": {"type": "mcp_tool_call_end"}})
    events.append(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": window,
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
            },
        }
    )
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


async def test_codex_backend_stream_forwards_cwd_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CodexBackend.stream delegates to the exec runner, forwarding cwd + the
    resumed session id (the cwd subprocess wiring itself is proven in A0)."""
    seen: dict[str, Any] = {}

    async def fake_exec(
        settings: Settings,
        prompt: str,
        session_id: str | None = None,
        image_paths: list[str] | None = None,
        *,
        cwd: str | None = None,
        sandbox: str = "read-only",
    ) -> AsyncIterator[StreamEvent]:
        seen["args"] = (prompt, session_id, image_paths, cwd)
        yield StreamDone(reply=AgentReply(text="ok"), session_id=session_id)

    monkeypatch.setattr("scufris.backends._stream_codex_exec", fake_exec)
    backend = CodexBackend("exec")
    events = [
        e
        async for e in backend.stream(Settings(), "hello", session_id="t1", cwd="/proj")
    ]
    assert seen["args"] == ("hello", "t1", None, "/proj")
    assert isinstance(events[-1], StreamDone)


async def test_codex_backend_app_server_mode_uses_app_server_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    used: dict[str, bool] = {}

    async def fake_app_server(
        settings: Settings,
        prompt: str,
        session_id: str | None = None,
        image_paths: list[str] | None = None,
        *,
        cwd: str | None = None,
        sandbox: str = "read-only",
    ) -> AsyncIterator[StreamEvent]:
        used["app_server"] = True
        yield StreamDone(reply=AgentReply(text="ok"))

    def fail_exec(*_a: Any, **_k: Any) -> AsyncIterator[StreamEvent]:
        raise AssertionError("exec runner must not be used in app_server mode")

    monkeypatch.setattr("scufris.backends._stream_app_server", fake_app_server)
    monkeypatch.setattr("scufris.backends._stream_codex_exec", fail_exec)
    backend = CodexBackend()  # defaults to the app_server runner
    _ = [e async for e in backend.stream(Settings(), "hi")]
    assert used.get("app_server") is True
    assert backend.name == "codex"  # friendly id, not the internal mode


def test_codex_backend_read_status_from_rollout(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    _write_rollout(home, "sess-1", cwd="/proj", turns=2, tools=1, last_answer="done")
    backend = CodexBackend("exec")
    settings = Settings(codex_home=home)

    status = backend.read_status(settings, "sess-1")
    assert status is not None
    assert status.session_id == "sess-1"
    assert status.turns == 2
    assert status.tool_calls == 1
    assert status.context_window == 258400
    assert status.output_tokens == 20
    assert status.last_message == "done"
    assert status.updated_at is not None

    # Unknown / missing session -> None (not an error).
    assert backend.read_status(settings, "nope") is None
    assert backend.read_status(settings, None) is None


async def test_mock_backend_stream_and_status() -> None:
    backend = MockBackend()
    events = [e async for e in backend.stream(Settings(), "ping", session_id="s")]
    assert isinstance(events[0], StreamTextDelta)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].session_id == "s"

    status = backend.read_status(Settings(), "s")
    assert status is not None
    assert status.session_id == "s"
    assert backend.read_status(Settings(), None) is None


def test_get_backend_normalizes_and_resolves() -> None:
    # Friendly ids resolve to their backend objects...
    assert isinstance(get_backend("codex"), CodexBackend)
    assert get_backend("codex").name == "codex"
    assert isinstance(get_backend("claude"), ClaudeBackend)
    assert isinstance(get_backend("mock"), MockBackend)
    # ...and legacy codex modes normalize to the codex backend.
    assert isinstance(get_backend("app_server"), CodexBackend)
    assert isinstance(get_backend("exec"), CodexBackend)
    assert get_backend("app_server").name == "codex"
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_backends_satisfy_the_protocol() -> None:
    assert isinstance(CodexBackend("exec"), AgentBackend)
    assert isinstance(MockBackend(), AgentBackend)
    assert isinstance(ClaudeBackend(), AgentBackend)
    # BackendStatus is a plain model with the normalized fields.
    st = BackendStatus(session_id="x")
    assert st.turns == 0 and st.last_message is None


# Faithful reductions of real `claude -p ... --output-format stream-json` output
# captured live (tasks/20260720-223938/NOTES.md).
_CLAUDE_SID = "982749a6-f764-4310-b6ab-092f3f74f48c"
_CLAUDE_STREAM_LINES = [
    json.dumps(
        {"type": "system", "subtype": "init", "session_id": _CLAUDE_SID, "cwd": "/x"}
    ),
    json.dumps(
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "PONG"}],
                "usage": {"input_tokens": 3989, "output_tokens": 5},
            },
            "session_id": _CLAUDE_SID,
        }
    ),
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "PONG",
            "session_id": _CLAUDE_SID,
            "num_turns": 1,
        }
    ),
]


def test_parse_claude_stream_from_probe() -> None:
    events = list(parse_claude_stream(_CLAUDE_STREAM_LINES))
    assert any(isinstance(e, StreamTextDelta) and e.delta == "PONG" for e in events)
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "PONG"
    assert done.session_id == _CLAUDE_SID


def test_parse_claude_stream_maps_tool_use_and_error() -> None:
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Bash"}]},
            }
        ),
        json.dumps({"type": "result", "subtype": "error", "result": "boom"}),
    ]
    events = list(parse_claude_stream(lines))
    assert any(isinstance(e, StreamTool) and e.tool.tool == "Bash" for e in events)
    assert isinstance(events[-1], StreamError)
    assert events[-1].detail == "boom"


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return (self._lines.pop(0) + "\n").encode()
        return b""


class _FakeProc:
    returncode = 0

    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)
        self.stderr = None

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:  # pragma: no cover - not reached on a clean exit
        pass


async def test_claude_backend_stream_parses_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio as _asyncio

    seen: dict[str, Any] = {}

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        seen["args"] = args
        seen["cwd"] = kwargs.get("cwd")
        return _FakeProc(_CLAUDE_STREAM_LINES)

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    backend = ClaudeBackend()
    settings = Settings(claude_bin="/usr/bin/true")
    events = [
        e async for e in backend.stream(settings, "ping", cwd="/proj", session_id="s1")
    ]
    assert seen["cwd"] == "/proj"
    assert "--resume" in seen["args"] and "s1" in seen["args"]
    assert "stream-json" in seen["args"]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].session_id == _CLAUDE_SID


def _write_claude_session(
    claude_home: Path, session_id: str, *, cwd_hash: str = "-proj"
) -> Path:
    proj = claude_home / "projects" / cwd_hash
    proj.mkdir(parents=True, exist_ok=True)
    path = proj / f"{session_id}.jsonl"
    lines = [
        {"type": "queue-operation", "operation": "x"},
        {"type": "user", "message": {"role": "user", "content": "do the thing"}},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash"},
                    {"type": "text", "text": "all done"},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 12},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_claude_backend_read_status_from_session(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    _write_claude_session(home, "sess-c1")
    backend = ClaudeBackend()
    settings = Settings(claude_home=home)

    status = backend.read_status(settings, "sess-c1")
    assert status is not None
    assert status.session_id == "sess-c1"
    assert status.turns == 1  # one string user turn (tool_result turns are lists)
    assert status.tool_calls == 1
    assert status.last_message == "all done"
    assert status.output_tokens == 12
    assert status.updated_at is not None

    assert backend.read_status(settings, "nope") is None
    assert backend.read_status(settings, None) is None


def test_parse_claude_transcript_maps_turns() -> None:
    """A claude session JSONL folds to user/assistant TranscriptMessages: string
    user turns kept, list (tool_result) user turns skipped, assistant turns
    concatenate text + carry tools/usage, tools-only assistant frames kept."""
    objs: list[dict[str, Any]] = [
        {"type": "queue-operation", "operation": "x"},
        {"type": "user", "message": {"content": "do the thing"}},
        {"type": "user", "message": {"content": ["tool_result"]}},  # skipped
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash"},
                    {"type": "text", "text": "all done"},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 12},
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read"}]},
        },
    ]
    msgs = parse_claude_transcript(objs)
    assert [m.role for m in msgs] == ["user", "assistant", "assistant"]
    assert msgs[0].text == "do the thing"
    assert msgs[1].text == "all done"
    assert [t.tool for t in msgs[1].tool_calls] == ["Bash"]
    assert msgs[1].usage is not None and msgs[1].usage.output_tokens == 12
    # A tools-only assistant frame is kept (no text).
    assert msgs[2].text == ""
    assert msgs[2].tool_calls[0].tool == "Read"


def test_claude_backend_read_transcript_from_session(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    _write_claude_session(home, "sess-t1")
    backend = ClaudeBackend()
    settings = Settings(claude_home=home)

    msgs = backend.read_transcript(settings, "sess-t1")
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].text == "do the thing"
    assert msgs[1].text == "all done"
    assert msgs[1].tool_calls[0].tool == "Bash"

    assert backend.read_transcript(settings, None) == []
    assert backend.read_transcript(settings, "nope") == []


def test_mock_backend_read_transcript_is_empty() -> None:
    assert MockBackend().read_transcript(Settings(), "sess") == []
    assert MockBackend().read_transcript(Settings(), None) == []


async def test_codex_backend_permission_mode_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each permission mode maps to the right codex --sandbox value."""
    seen: dict[str, Any] = {}

    async def fake_exec(
        settings: Settings,
        prompt: str,
        session_id: str | None = None,
        image_paths: list[str] | None = None,
        *,
        cwd: str | None = None,
        sandbox: str = "read-only",
    ) -> AsyncIterator[StreamEvent]:
        seen["sandbox"] = sandbox
        yield StreamDone(reply=AgentReply(text="ok"))

    monkeypatch.setattr("scufris.backends._stream_codex_exec", fake_exec)
    backend = CodexBackend("exec")
    for mode, expect in [
        ("manual", "read-only"),
        ("edit", "workspace-write"),
        ("auto", "danger-full-access"),
    ]:
        _ = [e async for e in backend.stream(Settings(), "go", permission_mode=mode)]
        assert seen["sandbox"] == expect
    # Default (no mode passed) is manual -> read-only.
    _ = [e async for e in backend.stream(Settings(), "go")]
    assert seen["sandbox"] == "read-only"


async def test_claude_backend_permission_mode_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio as _asyncio

    captured: list[tuple[Any, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.append(args)
        return _FakeProc(_CLAUDE_STREAM_LINES)

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    backend = ClaudeBackend()
    settings = Settings(claude_bin="/usr/bin/true")

    for mode, expect in [
        ("manual", "default"),
        ("edit", "acceptEdits"),
        ("auto", "bypassPermissions"),
    ]:
        captured.clear()
        _ = [e async for e in backend.stream(settings, "go", permission_mode=mode)]
        args = captured[0]
        assert "--permission-mode" in args
        assert expect in args
