"""Tests for the CLI dispatch."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from scufris import cli
from scufris.agent import AgentReply, AgentUnavailable, StreamDone, StreamEvent
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
    prints the terminal reply text - as an orchestrator turn carrying the
    orchestrator's configured write posture (default auto)."""
    seen: dict[str, object] = {}

    class FakeBackend:
        name = "fake"

        async def stream(
            self, settings: Settings, prompt: str, **kw: object
        ) -> AsyncIterator[StreamEvent]:
            seen.update(kw)
            yield StreamDone(reply=AgentReply(text=f"reply to {prompt}"))

    monkeypatch.setattr(cli, "get_backend", lambda name: FakeBackend())
    monkeypatch.setattr(cli, "run_server", lambda settings: None)
    monkeypatch.setenv("SCUFRIS_AGENT_ENABLED", "1")

    cli.main(["chat", "how are you"])
    assert "reply to how are you" in capsys.readouterr().out
    assert seen["is_orchestrator"] is True
    # One orchestrator, one posture: the CLI turn honours agent_permission_mode
    # (default auto) instead of silently running read-only.
    assert seen["permission_mode"] == "auto"


def test_chat_one_shot_stalled_turn_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-shot CLI runs outside the supervisor, so it supplies its own
    no-output backstop: a backend whose turn is idle-unbounded (opencode's
    read=None) must not hang the CLI forever. A stream that emits nothing within
    `agent_heartbeat_seconds` is cut with a clear error."""

    class StallBackend:
        name = "stall"

        async def stream(
            self, settings: Settings, prompt: str, **kw: object
        ) -> AsyncIterator[StreamEvent]:
            await asyncio.Event().wait()  # never fires: a genuinely stalled turn
            yield StreamDone(reply=AgentReply(text="unreachable"))

    monkeypatch.setattr(cli, "get_backend", lambda name: StallBackend())
    settings = Settings(agent_enabled=True, agent_heartbeat_seconds=0.2)
    with pytest.raises(AgentUnavailable, match="no output for 0.2s"):
        asyncio.run(cli._chat_once(settings, "hi"))


def test_hash_password_prints_a_verifiable_hash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`scufris hash-password` prints the env line the operator pastes into the
    sops dotenv, and the printed hash actually verifies the password.

    Asserts against `verify_password` (an INDEPENDENT check) rather than against
    another call to `hash_password`, which would pass even if both were broken -
    see lesson dod-named-tests-deserve-the-most-scrutiny."""
    from scufris.auth import verify_password

    answers = iter(["s3cret", "s3cret"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))

    cli.main(["hash-password"])

    out = capsys.readouterr().out.strip()
    assert out.startswith("SCUFRIS_AUTH_PASSWORD_HASH=")
    encoded = out.split("=", 1)[1]
    assert verify_password("s3cret", encoded)
    assert not verify_password("wrong", encoded)
    # The password itself must never be printed.
    assert "s3cret" not in out


def test_hash_password_refuses_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["one", "two"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hash-password"])
    assert "match" in str(excinfo.value)


def test_hash_password_refuses_an_empty_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
    with pytest.raises(SystemExit):
        cli.main(["hash-password"])


def test_serve_reports_a_fail_closed_auth_config_as_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LAN bind with no credential must exit with a readable message, not a
    traceback - the operator sees this in `journalctl`."""
    from scufris.auth import AuthConfigError

    def refuse(settings: Settings) -> None:
        raise AuthConfigError("no credential configured")

    monkeypatch.setattr(cli, "run_server", refuse)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["serve"])
    assert "refusing to start" in str(excinfo.value)
    assert "no credential configured" in str(excinfo.value)
