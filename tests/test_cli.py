"""Tests for the CLI dispatch."""

from __future__ import annotations

import pytest

from scufris import cli
from scufris.agent import AgentReply
from scufris.config import Settings


def test_no_subcommand_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[Settings] = []
    monkeypatch.setattr(cli, "run_server", lambda settings: called.append(settings))
    cli.main([])
    assert len(called) == 1


def test_serve_subcommand_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[Settings] = []
    monkeypatch.setattr(cli, "run_server", lambda settings: called.append(settings))
    cli.main(["serve"])
    assert len(called) == 1


def test_chat_subcommand_prints_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeAgent:
        async def chat(self, prompt: str) -> AgentReply:
            return AgentReply(text=f"reply to {prompt}")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(cli, "build_agent", lambda settings: FakeAgent())
    monkeypatch.setattr(cli, "run_server", lambda settings: None)

    cli.main(["chat", "how are you"])
    assert "reply to how are you" in capsys.readouterr().out
