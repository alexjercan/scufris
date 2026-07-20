"""Tests for the AgentBackend interface: CodexBackend (stream delegation +
rollout status), MockBackend, and the get_backend factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from scufris.agent import AgentReply, StreamDone, StreamEvent, StreamTextDelta
from scufris.backends import (
    AgentBackend,
    BackendStatus,
    CodexBackend,
    MockBackend,
    get_backend,
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
    ) -> AsyncIterator[StreamEvent]:
        used["app_server"] = True
        yield StreamDone(reply=AgentReply(text="ok"))

    def fail_exec(*_a: Any, **_k: Any) -> AsyncIterator[StreamEvent]:
        raise AssertionError("exec runner must not be used in app_server mode")

    monkeypatch.setattr("scufris.backends._stream_app_server", fake_app_server)
    monkeypatch.setattr("scufris.backends._stream_codex_exec", fail_exec)
    backend = CodexBackend("app_server")
    _ = [e async for e in backend.stream(Settings(), "hi")]
    assert used.get("app_server") is True
    assert backend.name == "app_server"


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


def test_get_backend_resolves_known_backends() -> None:
    assert get_backend("exec").name == "exec"
    assert get_backend("app_server").name == "app_server"
    assert get_backend("mock").name == "mock"
    # claude is not wired until A2b.
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("claude")


def test_backends_satisfy_the_protocol() -> None:
    assert isinstance(CodexBackend("exec"), AgentBackend)
    assert isinstance(MockBackend(), AgentBackend)
    # BackendStatus is a plain model with the normalized fields.
    st = BackendStatus(session_id="x")
    assert st.turns == 0 and st.last_message is None
