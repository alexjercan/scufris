"""Tests for settings parsing."""

from __future__ import annotations

import pytest

from scufris.config import Settings


def test_mcp_servers_default_empty() -> None:
    assert Settings().mcp_servers == []


def test_agent_defaults_enabled_app_server(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default to the streaming backend so the app does not silently fall back to
    # exec; the mock backend is available for offline dev/testing.
    monkeypatch.delenv("SCUFRIS_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("SCUFRIS_AGENT_BACKEND", raising=False)
    settings = Settings()
    assert settings.agent_enabled is True
    assert settings.agent_backend == "app_server"


def test_agent_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_BACKEND", "mock")
    assert Settings().agent_backend == "mock"


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
