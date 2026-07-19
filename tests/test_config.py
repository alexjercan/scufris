"""Tests for settings parsing."""

from __future__ import annotations

import pytest

from scufris.config import Settings


def test_mcp_servers_default_empty() -> None:
    assert Settings().mcp_servers == []


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
