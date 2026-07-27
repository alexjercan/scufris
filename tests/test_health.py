"""Tests for the operator-console health probes.

A fake ``codex_bin`` (a nonexistent path) makes the codex probes deterministic
and fast - ``create_subprocess_exec`` raises immediately, so no real codex is run.
The MCP tool count is the real in-process server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scufris.config import Settings
from scufris.enums import Backend
from scufris.health import agent_health


async def test_agent_health_reports_each_check(tmp_path: Path) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,  # probe the codex CLI (fake bin -> auth cannot be ok)
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
    )

    health = await agent_health(settings)

    by_name = {c.name: c for c in health.checks}
    assert by_name["agent"].status == "ok"
    # The orchestrator's MCP health is now one row PER server; the scufris agentic
    # server always advertises tools.
    assert by_name["mcp: scufris"].status == "ok"
    assert "tools ready" in by_name["mcp: scufris"].detail
    assert "mcp: den" in by_name  # the den life server gets its own row
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
        agent_backend=Backend.CLAUDE,
        agent_tools_enabled=True,
        claude_bin=str(tmp_path / "no-such-claude"),
    )

    health = await agent_health(settings)

    by_name = {c.name: c for c in health.checks}
    assert by_name["claude cli"].status == "warn"  # fake bin -> --version fails
    assert "codex cli" not in by_name
    assert "codex auth" not in by_name


async def test_agent_health_backend_param_overrides_global_backend(
    tmp_path: Path,
) -> None:
    """Passing ``backend=`` probes THAT backend regardless of the server's
    ``agent_backend`` - this is what makes a claude AGENT's health card show
    claude even when the orchestrator/server runs codex."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,  # the server default is codex...
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
        claude_bin=str(tmp_path / "no-such-claude"),
    )

    # ...but a claude agent's health probes claude, with no codex checks.
    health = await agent_health(settings, backend="claude")

    assert health.backend == "claude"
    by_name = {c.name: c for c in health.checks}
    assert by_name["claude cli"].status == "warn"
    assert "codex cli" not in by_name
    assert "codex auth" not in by_name
    # The neutral version field is populated from the probed backend (None here,
    # since the fake claude bin cannot report a version).
    assert health.backend_version is None


async def test_agent_health_backend_version_comes_from_the_probed_backend(
    tmp_path: Path,
) -> None:
    """A real (fake) claude bin that reports a version populates
    ``backend_version`` from THE CLAUDE probe - proving the neutral field carries
    the effective backend's version, not a codex string."""
    claude_bin = tmp_path / "claude"
    claude_bin.write_text('#!/bin/sh\necho "claude 9.9.9"\n')
    claude_bin.chmod(0o755)
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,  # server default is codex...
        agent_tools_enabled=True,
        claude_bin=str(claude_bin),
    )

    # ...but the claude agent's probe reports the claude version.
    health = await agent_health(settings, backend="claude")

    assert health.backend == "claude"
    assert health.backend_version == "claude 9.9.9"
    by_name = {c.name: c for c in health.checks}
    assert by_name["claude cli"].status == "ok"


async def test_agent_health_default_backend_is_the_server_backend(
    tmp_path: Path,
) -> None:
    """No ``backend=`` -> the server's ``agent_backend`` (orchestrator/global)."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
    )

    health = await agent_health(settings)

    assert health.backend == "codex"


async def test_agent_health_mcp_rows_are_per_server_and_audience_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator gets one MCP row per orchestrator server (scufris + den); a
    sub-agent gets only its callback server; a backend with no scufris MCP gets a
    single 'none' row."""
    (tmp_path / "den").mkdir()
    monkeypatch.setenv("SCUFRIS_DEN_PATH", str(tmp_path / "den"))
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
    )

    orch = await agent_health(settings, is_orchestrator=True)
    orch_mcp = {c.name for c in orch.checks if c.name.startswith("mcp")}
    assert orch_mcp == {"mcp: scufris", "mcp: den"}

    sub = await agent_health(settings, is_orchestrator=False)
    sub_mcp = {c.name for c in sub.checks if c.name.startswith("mcp")}
    assert sub_mcp == {"mcp: agent"}
    agent_row = next(c for c in sub.checks if c.name == "mcp: agent")
    assert agent_row.status == "ok" and "2 tools" in agent_row.detail

    none = await agent_health(settings, is_orchestrator=False, has_scufris_mcp=False)
    none_mcp = [c for c in none.checks if c.name.startswith("mcp")]
    assert len(none_mcp) == 1
    assert none_mcp[0].name == "mcp tools" and none_mcp[0].status == "warn"
    assert "no scufris tools" in none_mcp[0].detail


async def test_agent_health_den_row_warns_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no den configured, the den MCP row is amber (warn) - the operator sees
    which server is degraded, not just an aggregate."""
    monkeypatch.delenv("SCUFRIS_DEN_PATH", raising=False)
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
        _env_file=None,  # type: ignore[call-arg]
    )

    health = await agent_health(settings, is_orchestrator=True)
    by_name = {c.name: c for c in health.checks}
    assert by_name["mcp: scufris"].status == "ok"
    assert by_name["mcp: den"].status == "warn"
    assert "not configured" in by_name["mcp: den"].detail


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
