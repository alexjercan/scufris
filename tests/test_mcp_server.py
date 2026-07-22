"""Tests for the Scufris MCP tool server.

Tools are called directly (FastMCP's decorator returns the original function).
`host_stats` runs the real collector; the control tools call a respx-stubbed local
dashboard API, so the HTTP plumbing is exercised without a live server or the LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

from scufris.agent_store import AgentStore
from scufris.config import Settings
from scufris.mcp_server import (
    _agent_status_text,
    _format_processes,
    _list_agents_text,
    _run,
    apply_disabled_tools,
    create_agent,
    create_project,
    disk_usage,
    host_stats,
    list_processes,
    list_projects,
    mcp,
    message_agent,
    run_agent,
)
from scufris.processes import ProcessGroup, ProcessList
from scufris.projects import ProjectStore


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


def test_run_logs_the_command(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.DEBUG, logger="scufris.mcp_server"):
        _run([sys.executable, "-c", "print('hi', end='')"])
    assert any("exit=0" in record.getMessage() for record in caplog.records)


def test_main_configures_logging_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(mcp, "run", lambda: ran.append(True))
    from scufris.mcp_server import main as mcp_main

    mcp_main()
    assert ran == [True]


async def test_tools_registered() -> None:
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {
        "host_stats",
        "disk_usage",
        "list_processes",
        "list_agents",
        "agent_status",
        # orchestrator control tools (call the local dashboard API)
        "list_projects",
        "create_project",
        "create_agent",
        "run_agent",
        "message_agent",
    }
    assert all(tool.description for tool in await mcp.list_tools())


async def test_host_tool_descriptions_steer_away_from_shell() -> None:
    # The tool descriptions are one of the model's signals; they should explicitly
    # tell it to prefer these over raw shell (the real steering is the prompt
    # preamble in agent.py, but strong descriptions reinforce it).
    desc = {tool.name: (tool.description or "") for tool in await mcp.list_tools()}
    assert "PREFERRED" in desc["host_stats"] or "instead of shell" in desc["host_stats"]
    assert "uname" in desc["host_stats"] and "/proc" in desc["host_stats"]
    assert "PREFER" in desc["disk_usage"]
    assert "PREFER" in desc["list_processes"]


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


@pytest.fixture
def restore_tool_registry():
    """Snapshot and restore the module-level MCP tool registry.

    ``apply_disabled_tools`` mutates the process-global ``mcp`` singleton (fine
    in the real server subprocess, which is fresh per spawn), so tests that call
    it must restore the registry or they leak into later tests.
    """
    before = dict(mcp._tool_manager._tools)
    try:
        yield
    finally:
        mcp._tool_manager._tools = before


def test_apply_disabled_tools_removes_and_reports(restore_tool_registry) -> None:
    assert mcp._tool_manager.get_tool("disk_usage") is not None  # present before
    removed = apply_disabled_tools(["disk_usage", "does_not_exist"])
    assert removed == ["disk_usage"]  # only the real one reported
    # The disabled tool is gone from the live registry, so codex never sees it.
    assert mcp._tool_manager.get_tool("disk_usage") is None
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "disk_usage" not in names
    assert "host_stats" in names  # others untouched


def test_apply_disabled_tools_empty_is_noop(restore_tool_registry) -> None:
    before = {t.name for t in mcp._tool_manager.list_tools()}
    assert apply_disabled_tools([]) == []
    after = {t.name for t in mcp._tool_manager.list_tools()}
    assert before == after


# --- orchestrator observation tools ------------------------------------------


def _seed_agent(tmp_path: Path) -> tuple[Settings, AgentStore]:
    """A state dir with one project + one mock-backend agent."""
    settings = Settings(state_dir=tmp_path / "state", enable_mock_backend=True)
    proj = tmp_path / "proj"
    proj.mkdir()
    projects = ProjectStore(settings)
    projects.create(name="My App", cwd=str(proj))
    store = AgentStore(settings, projects)
    store.create(
        name="Builder", project_id="my-app", backend="mock", goal="do the thing"
    )
    return settings, store


def test_list_agents_text_formats_rows(tmp_path: Path) -> None:
    settings, _ = _seed_agent(tmp_path)
    out = _list_agents_text(settings)
    assert "ID" in out and "STATE" in out  # header
    assert "builder" in out
    assert "mock" in out
    assert "idle" in out


def test_list_agents_text_hides_the_reserved_orchestrator(tmp_path: Path) -> None:
    # The reserved orchestrator is a HIDDEN default - it is NOT in the agent list
    # (the tool lists the real, project-bound agents; the orchestrator is `/`).
    settings = Settings(state_dir=tmp_path / "state")
    out = _list_agents_text(settings)
    assert "orchestrator" not in out.lower()


def test_agent_status_text_reports_progress(tmp_path: Path) -> None:
    settings, store = _seed_agent(tmp_path)
    # A completed run persisted a session id + terminal state (cross-process: the
    # tool re-reads agents.json, so it sees what the run engine wrote).
    store.mark_finished("builder", state="done", session_id="mock-session")
    out = _agent_status_text(settings, "builder")
    assert "agent builder" in out
    assert "state: done" in out
    assert "backend: mock" in out
    assert "goal: do the thing" in out
    assert "mode: manual" in out
    # MockBackend.read_status -> turns=1, last_message "[mock] running".
    assert "turns: 1" in out
    assert "[mock] running" in out


def test_agent_status_text_unknown_id(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path / "state")
    assert "no such agent" in _agent_status_text(settings, "ghost")


# --- orchestrator control tools (call the local dashboard API) ---------------
#
# These wrap the dashboard's HTTP API; respx stubs it so each tool's method,
# path and body are asserted without a live server. Base URL is the tool's
# default (SCUFRIS_API_BASE unset -> http://127.0.0.1:8000).

_BASE = "http://127.0.0.1:8000"


@respx.mock
def test_list_projects_calls_api_and_formats() -> None:
    route = respx.get(f"{_BASE}/api/projects").mock(
        return_value=httpx.Response(
            200,
            json=[{"id": "p1", "name": "Web", "language": "python", "cwd": "/srv/web"}],
        )
    )
    out = list_projects()
    assert route.called
    assert "p1" in out and "Web" in out and "python" in out and "/srv/web" in out


@respx.mock
def test_list_projects_empty() -> None:
    respx.get(f"{_BASE}/api/projects").mock(return_value=httpx.Response(200, json=[]))
    assert "no projects" in list_projects()


@respx.mock
def test_create_project_posts_body_and_returns_result() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "p9", "name": "X", "cwd": "/tmp/x"})

    respx.post(f"{_BASE}/api/projects").mock(side_effect=handler)
    out = create_project("X", "/tmp/x")
    assert seen["body"] == {
        "name": "X",
        "cwd": "/tmp/x",
        "language": "",
        "description": "",
    }
    assert "p9" in out


def test_create_project_requires_name_and_cwd() -> None:
    # Guard runs before any HTTP call, so no respx route is needed.
    assert create_project("", "/tmp/x").startswith("error:")
    assert create_project("X", "  ").startswith("error:")


@respx.mock
def test_create_agent_posts_body_and_omits_empty_backend() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "ag1", "name": "A", "backend": "codex"})

    respx.post(f"{_BASE}/api/agents").mock(side_effect=handler)
    out = create_agent("A", "p1", backend="codex", permission_mode="edit")
    body = seen["body"]
    assert body["name"] == "A" and body["project_id"] == "p1"
    assert body["backend"] == "codex" and body["permission_mode"] == "edit"
    # model omitted -> not sent (server stamps its default)
    assert "model" not in body
    assert "ag1" in out


@respx.mock
def test_run_agent_posts_goal_and_returns_state() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"agent_id": "ag1", "state": "queued"})

    respx.post(f"{_BASE}/api/agents/ag1/run").mock(side_effect=handler)
    out = run_agent("ag1", goal="ship it")
    assert seen["body"] == {"goal": "ship it"}
    assert "queued" in out


@respx.mock
def test_message_agent_collects_sse_reply() -> None:
    # The chat endpoint streams SSE frames (`id:`/`data:` per _relay_bus_sse); the
    # tool collects the assistant reply from the done frame.
    sse = (
        'id: 1\ndata: {"kind":"text_delta","delta":"Hel"}\n\n'
        'id: 2\ndata: {"kind":"done","reply":{"text":"Hello there"}}\n\n'
    )
    respx.post(f"{_BASE}/api/agents/ag1/chat").mock(
        return_value=httpx.Response(200, text=sse)
    )
    assert message_agent("ag1", "hi") == "Hello there"


@respx.mock
def test_message_agent_reports_stream_error() -> None:
    sse = 'id: 1\ndata: {"kind":"error","detail":"boom"}\n\n'
    respx.post(f"{_BASE}/api/agents/ag1/chat").mock(
        return_value=httpx.Response(200, text=sse)
    )
    out = message_agent("ag1", "hi")
    assert out.startswith("error:") and "boom" in out


@respx.mock
def test_control_tool_error_path() -> None:
    # A non-2xx response yields an `error:` string carrying the code and detail,
    # never an exception.
    respx.post(f"{_BASE}/api/agents").mock(
        return_value=httpx.Response(422, text="unknown project")
    )
    out = create_agent("a", "nope")
    assert out.startswith("error:") and "422" in out and "unknown project" in out


@respx.mock
def test_control_tool_network_error_is_text() -> None:
    respx.get(f"{_BASE}/api/projects").mock(side_effect=httpx.ConnectError("refused"))
    out = list_projects()
    assert out.startswith("error:") and "failed" in out


def test_control_tool_rejects_bad_agent_id() -> None:
    # An id with a path or whitespace char is rejected before any HTTP call, so the
    # URL segment boundary is explicit (no respx route registered -> a call would 404
    # the mock and fail the test).
    assert run_agent("a/b").startswith("error:")
    assert run_agent("  ").startswith("error:")
    assert message_agent("a b", "hi").startswith("error:")
