"""Tests for the CLI dispatch."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from scufris import cli
from scufris.agent import AgentReply, StreamDone, StreamEvent
from scufris.config import Settings


def test_wants_debug_is_position_independent() -> None:
    assert cli._wants_debug(["serve"]) is False
    assert cli._wants_debug(["serve", "--debug"]) is True
    assert cli._wants_debug(["--debug", "serve"]) is True
    assert cli._wants_debug(["-v", "serve"]) is True


def test_debug_flag_is_accepted_in_both_positions() -> None:
    # argparse must not error on the flag before OR after the subcommand.
    parser = cli._build_parser()
    parser.parse_args(["--debug", "serve"])
    parser.parse_args(["serve", "--debug"])


def test_debug_flag_selects_debug_level(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(cli, "run_server", lambda settings: None)
    monkeypatch.setattr(
        cli,
        "configure_logging",
        lambda level, force=False: seen.append(level),
    )
    cli.main(["serve", "--debug"])
    cli.main(["serve"])
    assert seen == ["DEBUG", "INFO"]


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
    """The one-shot `chat` command drives the configured backend's stream and
    prints the terminal reply text."""

    class FakeBackend:
        name = "fake"

        async def stream(
            self, settings: Settings, prompt: str, **_kw: object
        ) -> AsyncIterator[StreamEvent]:
            yield StreamDone(reply=AgentReply(text=f"reply to {prompt}"))

    monkeypatch.setattr(cli, "get_backend", lambda name: FakeBackend())
    monkeypatch.setattr(cli, "run_server", lambda settings: None)
    monkeypatch.setenv("SCUFRIS_AGENT_ENABLED", "1")

    cli.main(["chat", "how are you"])
    assert "reply to how are you" in capsys.readouterr().out
