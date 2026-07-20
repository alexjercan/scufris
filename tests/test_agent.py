"""Tests for the agent backend.

The unit tests fake the `codex exec` runner seam; the integration tests point
``codex_bin`` at a tiny fake `codex` script, so the subprocess plumbing is
exercised for real without the actual codex binary or network.
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path
from typing import AsyncIterator

import pytest

from scufris.agent import (
    Agent,
    AgentReply,
    AgentUnavailable,
    CodexCliAgent,
    DisabledAgent,
    MockAgent,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamTool,
    TokenUsage,
    ToolCall,
    TurnOutcome,
    _appserver_event,
    _exec_args,
    _mcp_overrides,
    _parse_events,
    _run_codex_exec,
    _steer,
    _stream_app_server,
    _stream_codex_exec,
    build_agent,
)
from scufris.config import McpServerSpec, Settings
from scufris.sessions import STEERING_PREAMBLE, strip_steering


def _enabled(*, codex_bin: str | None = None, agent_model: str = "gpt-5.5") -> Settings:
    return Settings(agent_enabled=True, codex_bin=codex_bin, agent_model=agent_model)


def _write_fake_codex(path: Path, body: str) -> str:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_mcp_overrides_registers_scufris_by_default() -> None:
    args = _mcp_overrides(_enabled())
    joined = " ".join(args)
    assert "mcp_servers.scufris.command=" in joined
    assert "mcp_servers.scufris.args=" in joined
    assert 'mcp_servers.scufris.default_tools_approval_mode="approve"' in args
    assert 'approval_policy="never"' in args


def test_mcp_overrides_empty_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _mcp_overrides(settings) == []


def test_mcp_overrides_appends_configured_servers() -> None:
    settings = Settings(
        agent_enabled=True,
        mcp_servers=[
            McpServerSpec(
                id="fs", command="mcp-fs", args=["--root", "/tmp"], approve=False
            )
        ],
    )
    joined = " ".join(_mcp_overrides(settings))
    assert 'mcp_servers.fs.command="mcp-fs"' in joined
    assert "mcp_servers.fs.args=" in joined
    # approve=False -> no auto-approval line for this server
    assert "mcp_servers.fs.default_tools_approval_mode" not in joined


def test_mcp_overrides_skips_invalid_or_reserved_id() -> None:
    settings = Settings(
        agent_enabled=True,
        mcp_servers=[
            McpServerSpec(id="bad.id", command="x"),
            McpServerSpec(id="scufris", command="evil"),
        ],
    )
    joined = " ".join(_mcp_overrides(settings))
    assert "bad.id" not in joined  # invalid id skipped
    assert "evil" not in joined  # reserved scufris id not overridden


def test_mcp_overrides_passes_disabled_tools_env() -> None:
    settings = Settings(agent_enabled=True, disabled_tools=["tatr_new", "disk_usage"])
    joined = " ".join(_mcp_overrides(settings))
    assert "mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS=" in joined
    assert "tatr_new,disk_usage" in joined


def test_mcp_overrides_no_disabled_env_when_none() -> None:
    joined = " ".join(_mcp_overrides(_enabled()))
    assert "SCUFRIS_DISABLED_TOOLS" not in joined


def test_steer_prepends_preamble_when_tools_enabled() -> None:
    steered = _steer(_enabled(), "tell me about this host")
    assert steered.startswith(STEERING_PREAMBLE)
    assert steered.endswith("tell me about this host")
    # The preamble is transparently removable, so titles/transcripts stay clean.
    assert strip_steering(steered) == "tell me about this host"


def test_steer_noop_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _steer(settings, "hello") == "hello"


def test_exec_args_carries_steering_as_the_prompt() -> None:
    args = _exec_args("codex", _enabled(), "how full are my disks?", None, Path("/x"))
    # The prompt is the final arg and carries the steering preamble.
    assert args[-1].startswith(STEERING_PREAMBLE)
    assert strip_steering(args[-1]) == "how full are my disks?"


def test_exec_args_attaches_images() -> None:
    args = _exec_args(
        "codex", _enabled(), "look", None, Path("/x"), ["/tmp/a.png", "/tmp/b.jpg"]
    )
    # Each attached image rides as `--image <path>` before the prompt.
    assert "--image" in args
    joined = " ".join(args)
    assert "--image /tmp/a.png" in joined
    assert "--image /tmp/b.jpg" in joined
    assert args.index("--image") < args.index(args[-1])  # before the prompt


def test_build_agent_disabled_when_off() -> None:
    agent = build_agent(Settings(agent_enabled=False))
    assert isinstance(agent, DisabledAgent)
    assert isinstance(agent, Agent)


async def test_disabled_agent_chat_raises() -> None:
    agent = build_agent(Settings(agent_enabled=False))
    with pytest.raises(AgentUnavailable):
        await agent.chat("hello")
    await agent.aclose()


async def test_codex_cli_agent_uses_runner() -> None:
    seen: list[str] = []

    async def runner(
        _settings: Settings, prompt: str, _thread_id: str | None
    ) -> TurnOutcome:
        seen.append(prompt)
        return TurnOutcome(text=f"reply: {prompt}", thread_id="t1")

    agent = CodexCliAgent(_enabled(), runner=runner)
    assert isinstance(agent, Agent)

    reply = await agent.chat("hi")
    assert isinstance(reply, AgentReply)
    assert reply.text == "reply: hi"
    assert reply.status == "completed"
    assert seen == ["hi"]
    await agent.aclose()


async def test_codex_cli_agent_continues_and_resets_conversation() -> None:
    threads: list[str | None] = []

    async def runner(
        _settings: Settings, _prompt: str, thread_id: str | None
    ) -> TurnOutcome:
        threads.append(thread_id)
        return TurnOutcome(text="ok", thread_id="thread-123")

    agent = CodexCliAgent(_enabled(), runner=runner)
    await agent.chat("first")  # no thread yet
    await agent.chat("second")  # should resume the captured thread
    assert threads == [None, "thread-123"]

    agent.reset()
    await agent.chat("third")  # fresh conversation again
    assert threads == [None, "thread-123", None]


async def test_codex_cli_agent_switch_and_new_session() -> None:
    resumed: list[str | None] = []

    async def runner(
        _settings: Settings, _prompt: str, session_id: str | None
    ) -> TurnOutcome:
        resumed.append(session_id)
        return TurnOutcome(text="ok", thread_id="sess-A")

    agent = CodexCliAgent(_enabled(), runner=runner)
    assert agent.current_session_id() is None

    await agent.chat("hi")  # opens a session; id captured from the outcome
    assert agent.current_session_id() == "sess-A"

    agent.switch_session("sess-B")
    assert agent.current_session_id() == "sess-B"
    await agent.chat("continue")  # resumes the switched-to session
    assert resumed == [None, "sess-B"]

    agent.new_session()
    assert agent.current_session_id() is None


def test_disabled_agent_has_no_session() -> None:
    agent = DisabledAgent("off")
    assert agent.current_session_id() is None
    agent.switch_session("x")  # no-op, must not raise
    agent.new_session()
    assert agent.current_session_id() is None


async def test_build_agent_enabled_returns_codex_cli_agent() -> None:
    async def runner(
        _settings: Settings, _prompt: str, _thread_id: str | None
    ) -> TurnOutcome:
        return TurnOutcome(text="ok", thread_id=None)

    agent = build_agent(_enabled(), runner=runner)
    assert isinstance(agent, CodexCliAgent)


def test_parse_events_extracts_tools_and_usage() -> None:
    lines = [
        '{"type":"thread.started","thread_id":"t9"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"server":"scufris","tool":"host_stats","status":"completed"}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}',
        "not json",
        '{"type":"turn.completed","usage":{"input_tokens":14430,'
        '"cached_input_tokens":9984,"output_tokens":5,"reasoning_output_tokens":0}}',
    ]
    thread_id, tools, usage = _parse_events(("\n".join(lines) + "\n").encode())
    assert thread_id == "t9"
    assert len(tools) == 1
    assert tools[0].server == "scufris"
    assert tools[0].tool == "host_stats"
    assert tools[0].status == "completed"
    assert usage is not None
    assert usage.input_tokens == 14430
    assert usage.output_tokens == 5


async def test_chat_carries_tool_calls_and_usage() -> None:
    async def runner(
        _settings: Settings, _prompt: str, _thread_id: str | None
    ) -> TurnOutcome:
        return TurnOutcome(
            text="ok",
            thread_id="t1",
            tool_calls=[ToolCall(server="scufris", tool="tatr_ls", status="completed")],
            usage=TokenUsage(input_tokens=10, output_tokens=2),
        )

    reply = await CodexCliAgent(_enabled(), runner=runner).chat("hi")
    assert [t.tool for t in reply.tool_calls] == ["tatr_ls"]
    assert reply.usage is not None
    assert reply.usage.input_tokens == 10


async def test_run_codex_exec_reads_output_thread_tools_usage(tmp_path: Path) -> None:
    # A fake codex that emits thread.started + an mcp_tool_call + turn.completed
    # usage, and writes the final message.
    fake = _write_fake_codex(
        tmp_path / "codex",
        'out=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --output-last-message) out="$2"; shift 2;;\n'
        "    *) shift;;\n"
        "  esac\n"
        "done\n"
        'echo \'{"type":"thread.started","thread_id":"abc-123"}\'\n'
        'echo \'{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"server":"scufris","tool":"host_stats","status":"completed"}}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":100,'
        '"cached_input_tokens":10,"output_tokens":5,"reasoning_output_tokens":0}}\'\n'
        'printf "fake reply" > "$out"\n',
    )
    settings = _enabled(codex_bin=fake, agent_model="")
    outcome = await _run_codex_exec(settings, "hello")
    assert outcome.text == "fake reply"
    assert outcome.thread_id == "abc-123"
    assert [t.tool for t in outcome.tool_calls] == ["host_stats"]
    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 100


async def test_run_codex_exec_nonzero_exit_raises(tmp_path: Path) -> None:
    fake = _write_fake_codex(tmp_path / "codex", 'echo "boom" >&2\nexit 3\n')
    settings = _enabled(codex_bin=fake)
    with pytest.raises(AgentUnavailable, match="boom"):
        await _run_codex_exec(settings, "hello")


async def test_run_codex_exec_missing_binary_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scufris.agent.shutil.which", lambda _name: None)
    settings = _enabled(codex_bin=None)
    with pytest.raises(AgentUnavailable, match="codex CLI not found"):
        await _run_codex_exec(settings, "hello")


async def test_stream_codex_exec_emits_tool_then_done(tmp_path: Path) -> None:
    fake = _write_fake_codex(
        tmp_path / "codex",
        'out=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --output-last-message) out="$2"; shift 2;;\n'
        "    *) shift;;\n"
        "  esac\n"
        "done\n"
        'echo \'{"type":"thread.started","thread_id":"abc-123"}\'\n'
        'echo \'{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"server":"scufris","tool":"host_stats","status":"completed"}}\'\n'
        'echo \'{"type":"turn.completed","usage":{"input_tokens":100,'
        '"output_tokens":5}}\'\n'
        'printf "streamed reply" > "$out"\n',
    )
    settings = _enabled(codex_bin=fake, agent_model="")
    events: list[StreamEvent] = [e async for e in _stream_codex_exec(settings, "hi")]

    tools = [e for e in events if isinstance(e, StreamTool)]
    assert [t.tool.tool for t in tools] == ["host_stats"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "streamed reply"
    assert done.session_id == "abc-123"
    assert [t.tool for t in done.reply.tool_calls] == ["host_stats"]
    assert done.reply.usage is not None
    assert done.reply.usage.input_tokens == 100


async def test_stream_codex_exec_error_on_nonzero(tmp_path: Path) -> None:
    fake = _write_fake_codex(tmp_path / "codex", 'echo "boom" >&2\nexit 3\n')
    settings = _enabled(codex_bin=fake, agent_model="")
    events = [e async for e in _stream_codex_exec(settings, "hi")]
    last = events[-1]
    assert isinstance(last, StreamError)
    assert "boom" in last.detail


_TOOL_FAKE = (
    'out=""\n'
    'while [ $# -gt 0 ]; do case "$1" in --output-last-message) out="$2"; '
    "shift 2;; *) shift;; esac; done\n"
    'echo \'{"type":"thread.started","thread_id":"abc"}\'\n'
    'echo \'{"type":"item.completed","item":{"type":"mcp_tool_call",'
    '"server":"scufris","tool":"host_stats","status":"completed"}}\'\n'
    'echo \'{"type":"turn.completed","usage":{"input_tokens":100,'
    '"output_tokens":5}}\'\n'
    'printf "reply" > "$out"\n'
)


async def test_run_codex_exec_logs_the_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _write_fake_codex(tmp_path / "codex", _TOOL_FAKE)
    settings = _enabled(codex_bin=fake, agent_model="")
    long_prompt = "P" * 500
    with caplog.at_level(logging.DEBUG, logger="scufris.agent"):
        await _run_codex_exec(settings, long_prompt)
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "codex exec new" in blob
    assert "tool scufris.host_stats -> completed" in blob
    assert "usage input=100" in blob
    # The prompt is truncated, never logged in full.
    assert "P" * 500 not in blob
    assert "(+340 chars)" in blob


async def test_stream_codex_exec_logs_tools_and_events(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _write_fake_codex(tmp_path / "codex", _TOOL_FAKE)
    settings = _enabled(codex_bin=fake, agent_model="")
    with caplog.at_level(logging.DEBUG, logger="scufris.agent"):
        [e async for e in _stream_codex_exec(settings, "hi")]
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "codex json:" in blob  # each raw --json line at DEBUG
    assert "tool scufris.host_stats -> completed" in blob
    assert "codex exec stream new -> ok" in blob


async def test_chat_stream_updates_session_and_yields_events() -> None:
    async def stream_runner(
        _settings: Settings,
        prompt: str,
        _session_id: str | None,
        _image_paths: list[str] | None = None,
    ) -> "AsyncIterator[StreamEvent]":
        yield StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="completed")
        )
        yield StreamDone(
            reply=AgentReply(text=f"reply: {prompt}", status="completed"),
            session_id="sess-new",
        )

    agent = CodexCliAgent(_enabled(), stream_runner=stream_runner)
    events = [e async for e in agent.chat_stream("hi")]
    assert isinstance(events[0], StreamTool)
    assert isinstance(events[-1], StreamDone)
    assert events[-1].reply.text == "reply: hi"
    assert agent.current_session_id() == "sess-new"


async def test_disabled_agent_chat_stream_yields_error() -> None:
    agent = DisabledAgent("off")
    events = [e async for e in agent.chat_stream("hi")]
    assert len(events) == 1
    assert isinstance(events[0], StreamError)


# --- app-server backend ---


def test_appserver_event_maps_text_and_reasoning_deltas() -> None:
    text = _appserver_event(
        {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}}
    )
    assert isinstance(text, StreamTextDelta) and text.delta == "Hel"
    reason = _appserver_event(
        {"method": "item/reasoning/textDelta", "params": {"delta": "hmm"}}
    )
    assert isinstance(reason, StreamReasoningDelta) and reason.delta == "hmm"


def test_appserver_event_maps_tool_and_ignores_others() -> None:
    tool = _appserver_event(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "host_stats",
                    "status": "completed",
                }
            },
        }
    )
    assert isinstance(tool, StreamTool) and tool.tool.tool == "host_stats"
    assert _appserver_event({"method": "turn/started", "params": {}}) is None


_FAKE_APPSERVER = """#!/usr/bin/env python3
import sys, json
def out(o):
    sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line); rid = req.get("id"); m = req.get("method")
    if m == "initialize":
        out({"id": rid, "result": {}})
    elif m in ("thread/start", "thread/resume"):
        out({"id": rid, "result": {"thread": {"id": "t-1"}}})
    elif m == "turn/start":
        out({"id": rid, "result": {"turn": {}}})
        out({"method": "item/agentMessage/delta", "params": {"delta": "Hel"}})
        out({"method": "item/agentMessage/delta", "params": {"delta": "lo"}})
        out({"method": "thread/tokenUsage/updated",
             "params": {"tokenUsage": {"total": {"inputTokens": 5, "outputTokens": 2}}}})
        out({"method": "turn/completed", "params": {}})
        break
"""


async def test_stream_app_server_streams_text_deltas(tmp_path: Path) -> None:
    fake = tmp_path / "codex"
    fake.write_text(_FAKE_APPSERVER)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    settings = Settings(
        agent_enabled=True,
        codex_bin=str(fake),
        agent_model="",
        agent_tools_enabled=False,
    )
    events = [e async for e in _stream_app_server(settings, "hi")]

    deltas = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert deltas == ["Hel", "lo"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "Hello"
    assert done.session_id == "t-1"
    assert done.reply.usage is not None
    assert done.reply.usage.input_tokens == 5


def test_build_agent_selects_backend_stream_runner() -> None:
    app = build_agent(Settings(agent_enabled=True, agent_backend="app_server"))
    assert isinstance(app, CodexCliAgent)
    assert app._stream_runner is _stream_app_server
    # app_server is the default now.
    default = build_agent(Settings(agent_enabled=True))
    assert isinstance(default, CodexCliAgent)
    assert default._stream_runner is _stream_app_server
    exec_agent = build_agent(Settings(agent_enabled=True, agent_backend="exec"))
    assert isinstance(exec_agent, CodexCliAgent)
    assert exec_agent._stream_runner is _stream_codex_exec


def test_build_agent_mock_backend() -> None:
    agent = build_agent(Settings(agent_enabled=True, agent_backend="mock"))
    assert isinstance(agent, MockAgent)


async def test_mock_agent_streams_thinking_tool_and_tokens() -> None:
    agent = MockAgent()
    kinds: list[str] = []
    text = ""
    async for ev in agent.chat_stream("hello"):
        kinds.append(ev.kind)
        if ev.kind == "text_delta":
            text += ev.delta
    assert "reasoning_delta" in kinds
    assert "tool" in kinds
    assert kinds[-1] == "done"
    assert "mock" in text.lower()
    # The turn established a fake session that switch/reset drive.
    assert agent.current_session_id() is not None
    agent.reset()
    assert agent.current_session_id() is None
