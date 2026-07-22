"""Tests for settings parsing."""

from __future__ import annotations

import pytest

from scufris.config import Settings, auth_mode_for_backend


def test_mcp_servers_default_empty() -> None:
    assert Settings().mcp_servers == []


def test_auth_mode_for_backend_dispatches_by_backend() -> None:
    """The reported auth mode is per-backend: the codex mode for codex, the claude
    mode for claude, and None for a backend with no login (mock)."""
    settings = Settings()  # defaults: codex=chatgpt, claude=claude_ai
    assert auth_mode_for_backend(settings, "codex") == "chatgpt"
    assert auth_mode_for_backend(settings, "claude") == "claude_ai"
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
