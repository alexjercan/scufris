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

from scufris.mcp_server import (
    _format_processes,
    _run,
    disk_usage,
    host_stats,
    list_processes,
    mcp,
    tatr_ls,
    tatr_show,
)
from scufris.processes import ProcessGroup, ProcessList


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
    assert names == {
        "host_stats",
        "tatr_ls",
        "tatr_show",
        "disk_usage",
        "list_processes",
    }
    assert all(tool.description for tool in await mcp.list_tools())


def test_format_processes_renders_top_groups() -> None:
    plist = ProcessList(
        groups=[
            ProcessGroup(
                name="firefox",
                count=3,
                cpu_percent=42.5,
                mem_rss=3 * 1024 * 1024 * 1024,
                instances=[],
            ),
            ProcessGroup(
                name="python",
                count=1,
                cpu_percent=5.0,
                mem_rss=200 * 1024 * 1024,
                instances=[],
            ),
        ],
        total=57,
    )
    out = _format_processes(plist, limit=1)
    assert "APPLICATION" in out
    assert "total processes: 57" in out
    assert "firefox" in out
    assert "42.5" in out
    assert "3.0GB" in out
    assert "python" not in out  # limited to the top 1 group


def test_disk_usage_returns_table() -> None:
    out = disk_usage()
    # df -h prints a header row and at least the root filesystem.
    assert "Filesystem" in out
    assert "/" in out


def test_list_processes_returns_table() -> None:
    out = list_processes(limit=5)
    assert "APPLICATION" in out
    assert "total processes:" in out


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
