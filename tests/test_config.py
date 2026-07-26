"""Tests for settings parsing."""

from __future__ import annotations

import pytest

from scufris.config import Settings, auth_mode_for_backend


def test_mcp_servers_default_empty() -> None:
    assert Settings().mcp_servers == []


def test_telegram_defaults_off() -> None:
    settings = Settings()
    assert settings.telegram_bot_token is None
    assert settings.telegram_allowed_chat_ids == []


def test_telegram_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCUFRIS_TELEGRAM_BOT_TOKEN", "123:abc")
    assert Settings().telegram_bot_token == "123:abc"


def test_telegram_allowed_chat_ids_delimited_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comma/colon-separated env string parses to a list of ints."""
    monkeypatch.setenv("SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS", "123, 456:789")
    assert Settings().telegram_allowed_chat_ids == [123, 456, 789]


def test_telegram_allowed_chat_ids_json_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS", "[123, 456]")
    assert Settings().telegram_allowed_chat_ids == [123, 456]


def test_telegram_allowed_chat_ids_rejects_non_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS", "123,nope")
    with pytest.raises(ValueError):
        Settings()


def test_auth_mode_for_backend_dispatches_by_backend() -> None:
    """The reported auth mode is per-backend: the codex mode for codex, the claude
    mode for claude, and None for a backend with no login (mock)."""
    settings = Settings()  # defaults: codex=chatgpt, claude=claude_ai
    assert auth_mode_for_backend(settings, "codex") == "chatgpt"
    assert auth_mode_for_backend(settings, "claude") == "claude_ai"
    assert auth_mode_for_backend(settings, "opencode") == "local"  # self-hosted
    assert auth_mode_for_backend(settings, "mock") is None
    # Legacy codex mode ids fold to the codex auth mode.
    assert auth_mode_for_backend(settings, "app_server") == "chatgpt"


def test_auth_mode_for_backend_respects_api_key_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Either backend can be switched to api_key independently."""
    monkeypatch.setenv("SCUFRIS_AGENT_AUTH_MODE", "api_key")
    monkeypatch.setenv("SCUFRIS_AGENT_CLAUDE_AUTH_MODE", "api_key")
    settings = Settings()
    assert auth_mode_for_backend(settings, "codex") == "api_key"
    assert auth_mode_for_backend(settings, "claude") == "api_key"


def test_agent_defaults_enabled_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default to the codex backend; claude and mock (dev flag) are the other
    # options an agent - and the landing orchestrator - can be switched to.
    monkeypatch.delenv("SCUFRIS_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("SCUFRIS_AGENT_BACKEND", raising=False)
    settings = Settings()
    assert settings.agent_enabled is True
    assert settings.agent_backend == "codex"


def test_agent_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_BACKEND", "claude")
    assert Settings().agent_backend == "claude"
    monkeypatch.setenv("SCUFRIS_AGENT_BACKEND", "mock")
    assert Settings().agent_backend == "mock"


def test_legacy_backend_env_coerces_to_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pre-unification codex MODE ids still parse, folding to "codex".
    for legacy in ("app_server", "exec"):
        monkeypatch.setenv("SCUFRIS_AGENT_BACKEND", legacy)
        assert Settings().agent_backend == "codex"


def test_mcp_servers_parsed_from_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SCUFRIS_MCP_SERVERS",
        '[{"id": "files", "command": "mcp-fs", "args": ["--root", "/tmp"]}]',
    )
    settings = Settings()
    assert len(settings.mcp_servers) == 1
    server = settings.mcp_servers[0]
    assert server.id == "files"
    assert server.command == "mcp-fs"
    assert server.args == ["--root", "/tmp"]
    assert server.approve is True  # default
