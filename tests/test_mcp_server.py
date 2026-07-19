"""Tests for the Scufris MCP tool server.

Tools are called directly (FastMCP's decorator returns the original function).
`host_stats` runs the real collector; the `tatr_*` tools run the real `tatr`
against a temporary tasks dir, so the subprocess plumbing is exercised end to
end without the LLM.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scufris.mcp_server import _run, host_stats, mcp, tatr_ls, tatr_show


def test_host_stats_returns_snapshot() -> None:
    stats = host_stats()
    assert isinstance(stats, dict)
    assert stats["hostname"]
    assert "cpu_percent" in stats
    assert "mem" in stats


def test_run_reports_missing_binary() -> None:
    assert "not found on PATH" in _run(["scufris-no-such-binary-xyz"])


def test_run_captures_stdout() -> None:
    assert _run([sys.executable, "-c", "print('hi', end='')"]) == "hi"


def test_run_reports_nonzero_exit() -> None:
    out = _run(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(2)"]
    )
    assert "boom" in out


async def test_tools_registered() -> None:
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {"host_stats", "tatr_ls", "tatr_show"}


def _new_task(cwd: Path, title: str) -> str:
    (cwd / "tasks").mkdir(exist_ok=True)
    result = subprocess.run(
        ["tatr", "new", title],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    # "Task created successfully with ID: YYYYMMDD-HHMMSS"
    return result.stdout.strip().split()[-1]


def test_tatr_ls_lists_created_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _new_task(tmp_path, "Hello MCP task")
    monkeypatch.chdir(tmp_path)
    assert "Hello MCP task" in tatr_ls()


def test_tatr_show_shows_task_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = _new_task(tmp_path, "Show me")
    monkeypatch.chdir(tmp_path)
    assert "Show me" in tatr_show(task_id)
