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
    _claude_stream_args,
    get_backend,
    parse_claude_stream,
    parse_claude_transcript,
)
from scufris.config import Settings
from scufris.reasoning_store import ReasoningStore


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
    """CodexBackend.stream delegates to the app_server runner, forwarding cwd +
    the resumed session id (the cwd subprocess wiring itself is proven in A0)."""
    seen: dict[str, Any] = {}

    async def fake_exec(
        settings: Settings,
        prompt: str,
        session_id: str | None = None,
        image_paths: list[str] | None = None,
        *,
        cwd: str | None = None,
        sandbox: str = "read-only",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        seen["args"] = (prompt, session_id, image_paths, cwd)
        yield StreamDone(reply=AgentReply(text="ok"), session_id=session_id)

    monkeypatch.setattr("scufris.backends._stream_app_server", fake_exec)
    backend = CodexBackend()
    events = [
        e
        async for e in backend.stream(Settings(), "hello", session_id="t1", cwd="/proj")
    ]
    assert seen["args"] == ("hello", "t1", None, "/proj")
    assert isinstance(events[-1], StreamDone)


async def test_codex_backend_uses_app_server_runner(
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
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        used["app_server"] = True
        yield StreamDone(reply=AgentReply(text="ok"))

    monkeypatch.setattr("scufris.backends._stream_app_server", fake_app_server)
    backend = CodexBackend()  # always the app_server runner (exec mode dropped)
    _ = [e async for e in backend.stream(Settings(), "hi")]
    assert used.get("app_server") is True
    assert backend.name == "codex"  # friendly id


def test_codex_backend_read_status_from_rollout(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    _write_rollout(home, "sess-1", cwd="/proj", turns=2, tools=1, last_answer="done")
    backend = CodexBackend()
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


def test_codex_backend_read_transcript_merges_sidecar_reasoning(
    tmp_path: Path,
) -> None:
    # A rollout carries the answers but NOT the reasoning (encrypted on disk);
    # the sidecar under state_dir carries the "thinking". read_transcript must
    # merge them so a reloaded transcript's assistant turns show the spoiler.
    home = tmp_path / "codex"
    _write_rollout(home, "sess-1", cwd="/proj", turns=1, last_answer="the answer")
    backend = CodexBackend()
    settings = Settings(codex_home=home, state_dir=tmp_path / "state")

    ReasoningStore(settings).append("sess-1", "I thought hard", answer="the answer")

    messages = backend.read_transcript(settings, "sess-1")
    assistant = [m for m in messages if m.role == "assistant"]
    assert assistant[0].reasoning == "I thought hard"


def test_codex_backend_read_transcript_without_sidecar_is_graceful(
    tmp_path: Path,
) -> None:
    # A pre-existing session (no sidecar) renders normally: no reasoning, no error.
    home = tmp_path / "codex"
    _write_rollout(home, "sess-1", cwd="/proj", turns=1, last_answer="hi")
    backend = CodexBackend()
    settings = Settings(codex_home=home, state_dir=tmp_path / "state")

    messages = backend.read_transcript(settings, "sess-1")
    assert all(m.reasoning is None for m in messages)


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
    assert isinstance(CodexBackend(), AgentBackend)
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio as _asyncio

    seen: dict[str, Any] = {}

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        seen["args"] = args
        seen["cwd"] = kwargs.get("cwd")
        return _FakeProc(_CLAUDE_STREAM_LINES)

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    backend = ClaudeBackend()
    # The session must exist on disk for --resume to be passed (else the backend
    # starts fresh - see _claude_stream_args), so seed it under a temp home.
    home = tmp_path / "claude"
    _write_claude_session(home, "s1")
    settings = Settings(claude_bin="/usr/bin/true", claude_home=home)
    events = [
        e async for e in backend.stream(settings, "ping", cwd="/proj", session_id="s1")
    ]
    assert seen["cwd"] == "/proj"
    assert "--resume" in seen["args"] and "s1" in seen["args"]
    assert "stream-json" in seen["args"]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].session_id == _CLAUDE_SID


def test_claude_stream_mints_session_id(tmp_path: Path) -> None:
    """A FRESH claude turn (no resumable session) passes a scufris-minted
    --session-id so the id is deterministic and known before the turn, instead
    of being read back from the result frame. --resume is not used."""
    home = tmp_path / "claude"  # empty -> None is unresumable
    args = _claude_stream_args(
        "claude", "hi", "manual", None, home, Settings(), new_session_id="fixed-uuid-1"
    )
    assert "--resume" not in args
    i = args.index("--session-id")
    assert args[i + 1] == "fixed-uuid-1"


def test_claude_stream_resumes_existing_session(tmp_path: Path) -> None:
    """Resume wins over mint: an on-disk session uses --resume and NEVER
    --session-id (passing both would be contradictory), even when a fresh id is
    offered."""
    home = tmp_path / "claude"
    _write_claude_session(home, "on-disk")
    args = _claude_stream_args(
        "claude", "hi", "manual", "on-disk", home, Settings(), new_session_id="ignored"
    )
    assert args[-2:] == ["--resume", "on-disk"]
    assert "--session-id" not in args


async def test_claude_stream_done_carries_minted_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On a fresh turn the backend mints a UUID, passes it as --session-id, and
    guarantees StreamDone carries it EVEN WHEN the result frame omits session_id
    (a turn that dies before a full result no longer orphans its session)."""
    import asyncio as _asyncio
    import uuid as _uuid

    # A stream whose result frame has NO session_id.
    lines = [
        json.dumps({"type": "system", "subtype": "init", "cwd": "/x"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            }
        ),
        json.dumps(
            {"type": "result", "subtype": "success", "is_error": False, "result": "hi"}
        ),
    ]
    seen: dict[str, Any] = {}

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        seen["args"] = args
        return _FakeProc(lines)

    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    backend = ClaudeBackend()
    settings = Settings(claude_bin="/usr/bin/true", claude_home=tmp_path / "claude")
    events = [
        e async for e in backend.stream(settings, "ping", cwd="/proj", session_id=None)
    ]
    argv = seen["args"]
    assert "--resume" not in argv
    minted = argv[argv.index("--session-id") + 1]
    _uuid.UUID(minted)  # a real UUID
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.session_id == minted  # substituted despite the frame omitting it


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


def test_claude_backend_read_context_maps_status(tmp_path: Path) -> None:
    """claude has no per-session context window, so read_context maps its status
    snapshot: a real turn/token count, window 0."""
    home = tmp_path / "claude"
    _write_claude_session(home, "sess-c1")
    ctx = ClaudeBackend().read_context(Settings(claude_home=home), "sess-c1")
    assert ctx is not None
    assert ctx.session_id == "sess-c1"
    assert ctx.turn_count == 1
    assert ctx.tool_call_count == 1
    assert ctx.input_tokens == 200
    assert ctx.context_window == 0  # claude does not expose a window
    # Unknown/unset -> None.
    assert ClaudeBackend().read_context(Settings(claude_home=home), None) is None
    assert ClaudeBackend().read_context(Settings(claude_home=home), "nope") is None


async def test_claude_backend_delete_session_unlinks_file(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    path = _write_claude_session(home, "sess-d1")
    backend = ClaudeBackend()
    settings = Settings(claude_home=home)
    assert path.exists()
    assert await backend.delete_session(settings, "sess-d1") is True
    assert not path.exists()
    # Idempotent: a second delete (now gone) is False, never raises.
    assert await backend.delete_session(settings, "sess-d1") is False
    assert await backend.delete_session(settings, None) is False


async def test_mock_backend_context_and_delete_are_noops() -> None:
    assert MockBackend().read_context(Settings(), "sess") is None
    assert await MockBackend().delete_session(Settings(), "sess") is False


def test_claude_stream_skips_unresumable_session(tmp_path: Path) -> None:
    """--resume is passed only for a session that exists on disk; an unknown id
    (stale/deleted/cross-backend) starts fresh instead of failing the turn with
    error_during_execution. Regression pin for 20260721-152034."""
    home = tmp_path / "claude"
    _write_claude_session(home, "on-disk")
    settings = Settings()

    # No session id -> no --resume.
    assert "--resume" not in _claude_stream_args(
        "claude", "hi", "manual", None, home, settings
    )

    # A session that exists -> --resume it.
    args = _claude_stream_args("claude", "hi", "manual", "on-disk", home, settings)
    assert args[-2:] == ["--resume", "on-disk"]

    # A session NOT on disk (e.g. a codex id after a backend switch) -> omit
    # --resume so claude starts a fresh conversation.
    assert "--resume" not in _claude_stream_args(
        "claude", "hi", "manual", "codex-uuid", home, settings
    )


def _claude_server_config(args: list[str], server_id: str) -> dict[str, Any]:
    """The inline server dict for ``server_id`` parsed out of a claude argv's
    --mcp-config (the config now carries one entry per registered server)."""
    i = args.index("--mcp-config")
    config = json.loads(args[i + 1])
    return config["mcpServers"][server_id]  # type: ignore[no-any-return]


def _hermetic() -> Settings:
    # Ignore the repo `.env` so a dev box's SCUFRIS_DEN_PATH does not add a `den`
    # server to turns these tests assert the shape of.
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_claude_stream_args_wires_scufris_for_orchestrator(tmp_path: Path) -> None:
    """The claude argv registers the scufris server: an inline --mcp-config,
    --strict-mcp-config, and the whole-server allowlist so the unattended turn does
    not hang on approval. No den configured -> scufris only, no callback server."""
    args = _claude_stream_args(
        "claude",
        "hi",
        "manual",
        None,
        tmp_path / "c",
        _hermetic(),
        is_orchestrator=True,
    )
    assert "--strict-mcp-config" in args
    allowed = [args[i + 1] for i, a in enumerate(args) if a == "--allowedTools"]
    assert allowed == ["mcp__scufris__*"]
    server = _claude_server_config(args, "scufris")
    assert server["args"] == ["-m", "scufris.mcp_server"]
    assert server["env"]["SCUFRIS_API_BASE"].startswith("http://")
    # The role env is retired (the audience split is physical).
    assert "SCUFRIS_AGENT_ROLE" not in server["env"]
    # The variadic --mcp-config MUST be bounded by a following flag, never a
    # positional (else it swallows later argv tokens as config paths).
    j = args.index("--mcp-config")
    assert args[j + 2].startswith("--")


def test_claude_stream_args_registers_den_for_orchestrator(tmp_path: Path) -> None:
    """With a den configured, the orchestrator turn also registers the `den` server
    with its own allowlist and carries the den path only on it."""
    settings = Settings(den_path=Path("/home/op/the-den"), _env_file=None)  # type: ignore[call-arg]
    args = _claude_stream_args(
        "claude", "hi", "manual", None, tmp_path / "c", settings, is_orchestrator=True
    )
    allowed = [args[i + 1] for i, a in enumerate(args) if a == "--allowedTools"]
    assert allowed == ["mcp__scufris__*", "mcp__den__*"]
    den = _claude_server_config(args, "den")
    assert den["args"] == ["-m", "scufris.den_mcp_server"]
    assert den["env"]["SCUFRIS_DEN_PATH"] == "/home/op/the-den"


def test_claude_stream_args_agent_registers_only_callback_server(
    tmp_path: Path,
) -> None:
    args = _claude_stream_args(
        "claude", "hi", "manual", None, tmp_path / "c", _hermetic(), agent_id="builder"
    )
    allowed = [args[i + 1] for i, a in enumerate(args) if a == "--allowedTools"]
    assert allowed == ["mcp__agent__*"]
    server = _claude_server_config(args, "agent")
    assert server["args"] == ["-m", "scufris.agent_mcp_server"]
    assert server["env"]["SCUFRIS_AGENT_ID"] == "builder"
    assert "SCUFRIS_AGENT_ROLE" not in server["env"]


def test_claude_stream_args_passes_disabled_tools(tmp_path: Path) -> None:
    settings = Settings(disabled_tools=["run_agent", "message_agent"], _env_file=None)  # type: ignore[call-arg]
    args = _claude_stream_args(
        "claude", "hi", "manual", None, tmp_path / "c", settings, is_orchestrator=True
    )
    env = _claude_server_config(args, "scufris")["env"]
    assert env["SCUFRIS_DISABLED_TOOLS"] == "run_agent,message_agent"


def test_claude_stream_args_no_mcp_when_disabled_or_no_audience(tmp_path: Path) -> None:
    home = tmp_path / "c"
    # Tools disabled -> no scufris flags even for the orchestrator.
    off = _claude_stream_args(
        "claude",
        "hi",
        "manual",
        None,
        home,
        Settings(agent_tools_enabled=False),
        is_orchestrator=True,
    )
    assert "--mcp-config" not in off
    # Enabled but no audience (a plain turn) -> still no scufris server.
    plain = _claude_stream_args("claude", "hi", "manual", None, home, _hermetic())
    assert "--mcp-config" not in plain


def test_claude_stream_args_keeps_mcp_config_on_resume(tmp_path: Path) -> None:
    """Args are rebuilt each turn, so a RESUMED turn re-loads --mcp-config the same
    as a fresh one (like codex re-sends its sandbox per turn) - and the resume flag
    does not break the variadic --mcp-config's flag boundary."""
    home = tmp_path / "claude"
    _write_claude_session(home, "on-disk")
    args = _claude_stream_args(
        "claude", "hi", "manual", "on-disk", home, Settings(), is_orchestrator=True
    )
    assert "--mcp-config" in args
    assert args[-2:] == ["--resume", "on-disk"]
    j = args.index("--mcp-config")
    assert args[j + 2].startswith("--")


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
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        seen["sandbox"] = sandbox
        yield StreamDone(reply=AgentReply(text="ok"))

    monkeypatch.setattr("scufris.backends._stream_app_server", fake_exec)
    backend = CodexBackend()
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

    # stream() threads is_orchestrator/agent_id through to the scufris wiring: an
    # orchestrator turn's argv registers the scufris server, a plain turn does not.
    captured.clear()
    _ = [e async for e in backend.stream(settings, "go", is_orchestrator=True)]
    assert "--mcp-config" in captured[0]
    assert "mcp__scufris__*" in captured[0]

    captured.clear()
    _ = [e async for e in backend.stream(settings, "go")]
    assert "--mcp-config" not in captured[0]
