"""Tests for the agent backend.

The unit tests fake the `codex exec` runner seam; the integration tests point
``codex_bin`` at a tiny fake `codex` script, so the subprocess plumbing is
exercised for real without the actual codex binary or network.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from scufris.agent import (
    Agent,
    AgentReply,
    AgentUnavailable,
    CodexCliAgent,
    DisabledAgent,
    TokenUsage,
    ToolCall,
    TurnOutcome,
    _mcp_overrides,
    _parse_events,
    _run_codex_exec,
    build_agent,
)
from scufris.config import McpServerSpec, Settings


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
