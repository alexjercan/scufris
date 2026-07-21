"""Tests for the operator-console health probes.

A fake ``codex_bin`` (a nonexistent path) makes the codex probes deterministic
and fast - ``create_subprocess_exec`` raises immediately, so no real codex is run.
The MCP tool count is the real in-process server.
"""

from __future__ import annotations

from pathlib import Path

from scufris.config import Settings
from scufris.health import agent_health


async def test_agent_health_reports_each_check(tmp_path: Path) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend="codex",  # probe the codex CLI (fake bin -> auth cannot be ok)
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
    )

    health = await agent_health(settings)

    by_name = {c.name: c for c in health.checks}
    assert by_name["agent"].status == "ok"
    # The real in-process MCP server exposes tools.
    assert by_name["mcp tools"].status == "ok"
    assert "tools available" in by_name["mcp tools"].detail
    # web/dist is absent in the temp path.
    assert by_name["web assets"].status == "error"
    assert by_name["web assets"].hint
    # Versions are populated (scufris always has metadata).
    assert health.scufris_version
    # codex is not really run (fake bin), so auth cannot be "ok".
    assert by_name["codex auth"].status == "warn"


async def test_agent_health_probes_claude_backend(tmp_path: Path) -> None:
    """A claude-backed orchestrator probes the CLAUDE cli, not codex: a broken
    claude binary warns, and no codex checks are emitted."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend="claude",
        agent_tools_enabled=True,
        claude_bin=str(tmp_path / "no-such-claude"),
    )

    health = await agent_health(settings)

    by_name = {c.name: c for c in health.checks}
    assert by_name["claude cli"].status == "warn"  # fake bin -> --version fails
    assert "codex cli" not in by_name
    assert "codex auth" not in by_name


async def test_agent_health_flags_disabled_agent_and_tools(tmp_path: Path) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=False,
        agent_tools_enabled=False,
        codex_bin=str(tmp_path / "no-such-codex"),
    )

    health = await agent_health(settings)

    by_name = {c.name: c.status for c in health.checks}
    assert by_name["agent"] == "warn"
    assert by_name["mcp tools"] == "warn"
    assert health.session_count == 0
