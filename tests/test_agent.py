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
    TurnOutcome,
    _run_codex_exec,
    build_agent,
)
from scufris.config import Settings


def _enabled(*, codex_bin: str | None = None, agent_model: str = "gpt-5.5") -> Settings:
    return Settings(agent_enabled=True, codex_bin=codex_bin, agent_model=agent_model)


def _write_fake_codex(path: Path, body: str) -> str:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


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


async def test_build_agent_enabled_returns_codex_cli_agent() -> None:
    async def runner(
        _settings: Settings, _prompt: str, _thread_id: str | None
    ) -> TurnOutcome:
        return TurnOutcome(text="ok", thread_id=None)

    agent = build_agent(_enabled(), runner=runner)
    assert isinstance(agent, CodexCliAgent)


async def test_run_codex_exec_reads_output_and_thread_id(tmp_path: Path) -> None:
    # A fake codex that emits a thread.started event and writes the final message.
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
        'printf "fake reply" > "$out"\n',
    )
    settings = _enabled(codex_bin=fake, agent_model="")
    outcome = await _run_codex_exec(settings, "hello")
    assert outcome.text == "fake reply"
    assert outcome.thread_id == "abc-123"


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
