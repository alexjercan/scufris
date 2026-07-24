"""Tests for the Scufris MCP tool server.

Tools are called directly (FastMCP's decorator returns the original function).
`host_stats` runs the real collector; the control tools call a respx-stubbed local
dashboard API, so the HTTP plumbing is exercised without a live server or the LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from scufris.agent_store import AgentStore
from scufris.config import Settings
from scufris.enums import AgentState
from scufris.mcp_server import (
    ROLE_AGENT,
    ROLE_ORCHESTRATOR,
    _agent_status_text,
    _format_processes,
    _list_agents_text,
    _run,
    acknowledge,
    apply_disabled_tools,
    apply_role,
    create_agent,
    create_project,
    delete_agent,
    delete_project,
    disk_usage,
    get_project,
    host_stats,
    list_processes,
    list_projects,
    mcp,
    message_agent,
    pending_agents,
    request_input,
    run_agent,
    update_agent,
    update_project,
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


def test_main_configures_logging_and_runs(
    monkeypatch: pytest.MonkeyPatch, restore_tool_registry
) -> None:
    # main() now role-scopes the live registry (apply_role) before serving, so it
    # must be restored or it leaks a trimmed tool set into later tests.
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
        "get_project",
        "create_project",
        "update_project",
        "delete_project",
        "create_agent",
        "update_agent",
        "delete_agent",
        "run_agent",
        "message_agent",
        # orchestrator-side agent-comms tools (BC3)
        "pending_agents",
        "acknowledge",
        # sub-agent callback tool (agent-role only; role-scoped at startup)
        "request_input",
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


# --- BC2: role-scoped tool exposure -------------------------------------------


def test_apply_role_agent_keeps_only_request_input(restore_tool_registry) -> None:
    """A sub-agent role exposes ONLY request_input - no control/observe/host
    tools reach a regular agent (the guarantee T3 cares about)."""
    apply_role(ROLE_AGENT)
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert names == {"request_input"}


def test_apply_role_orchestrator_drops_only_the_agent_tools(
    restore_tool_registry,
) -> None:
    """The orchestrator role keeps the full surface but drops the agent-only
    callback tools (it is the recipient of request_input, not a caller)."""
    removed = apply_role(ROLE_ORCHESTRATOR)
    assert removed == ["request_input"]
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "request_input" not in names
    assert {"host_stats", "create_agent", "run_agent", "list_agents"} <= names


@respx.mock
def test_request_input_posts_question_for_the_env_agent(monkeypatch) -> None:
    """request_input addresses the caller's own id (SCUFRIS_AGENT_ID) and posts
    the question to the request_input endpoint."""
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"agent_id": "builder", "state": "waiting"})

    route = respx.post(f"{_BASE}/api/agents/builder/request_input").mock(
        side_effect=handler
    )
    out = request_input("should I merge to master?")
    assert route.called
    assert seen["body"] == {"question": "should I merge to master?"}
    assert "waiting" in out


def test_request_input_without_agent_id_is_an_error(monkeypatch) -> None:
    """With no SCUFRIS_AGENT_ID in the environment, request_input refuses rather
    than posting to a bogus path."""
    monkeypatch.delenv("SCUFRIS_AGENT_ID", raising=False)
    out = request_input("merge?")
    assert out.startswith("error:")


def test_request_input_requires_a_question(monkeypatch) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    assert request_input("   ").startswith("error:")


@respx.mock
def test_pending_agents_formats_the_poll() -> None:
    """pending_agents GETs /api/agents/pending and renders a row per waiter (BC3)."""
    respx.get("http://127.0.0.1:8000/api/agents/pending").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "agent_id": "builder",
                    "state": "waiting",
                    "message": "should I merge to master?",
                    "run_id": "builder:r1",
                    "session_id": "s1",
                    "ts": 1.0,
                }
            ],
        )
    )
    out = pending_agents()
    assert "builder" in out and "waiting" in out and "merge to master" in out


@respx.mock
def test_pending_agents_empty() -> None:
    respx.get("http://127.0.0.1:8000/api/agents/pending").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert "no agents are waiting" in pending_agents()


@respx.mock
def test_acknowledge_posts_to_the_endpoint() -> None:
    route = respx.post("http://127.0.0.1:8000/api/agents/builder/acknowledge").mock(
        return_value=httpx.Response(
            200, json={"agent_id": "builder", "acknowledged": True}
        )
    )
    out = acknowledge("builder")
    assert route.called
    assert "acknowledged" in out


def test_acknowledge_rejects_a_bad_id() -> None:
    assert acknowledge("a/b").startswith("error:")


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
    store.mark_finished("builder", state=AgentState.DONE, session_id="mock-session")
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
    seen: dict[str, Any] = {}

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
def test_message_agent_read_timeout_is_unbounded() -> None:
    """Steering a sub-agent runs a full turn that streams SSE until it finishes; a
    long-but-progressing turn (or a long silent tool call) must not be cut by a
    wall-clock read cap. The chat request disables the read timeout (the turn
    self-terminates: runner idle guard + supervisor heartbeat), while connect
    stays bounded so an unreachable API still fails fast."""
    sse = 'id: 1\ndata: {"kind":"done","reply":{"text":"ok"}}\n\n'
    route = respx.post(f"{_BASE}/api/agents/ag1/chat").mock(
        return_value=httpx.Response(200, text=sse)
    )
    assert message_agent("ag1", "hi") == "ok"
    timeout = route.calls[0].request.extensions["timeout"]
    assert timeout["read"] is None
    assert timeout["connect"] == 15.0  # _API_TIMEOUT still bounds connect


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


# --- CRUD control tools: get/update/delete project; update/delete agent ------


@respx.mock
def test_get_project_calls_endpoint() -> None:
    route = respx.get(f"{_BASE}/api/projects/p1").mock(
        return_value=httpx.Response(200, json={"id": "p1", "name": "Web"})
    )
    out = get_project("p1")
    assert route.called
    assert "Web" in out


@respx.mock
def test_update_project_patches_only_provided_fields() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "p1", "language": "rust"})

    respx.patch(f"{_BASE}/api/projects/p1").mock(side_effect=handler)
    out = update_project("p1", language="rust")
    # Only the provided field is sent (ProjectUpdate is extra=forbid).
    assert seen["body"] == {"language": "rust"}
    assert "rust" in out


def test_update_project_requires_a_field() -> None:
    # No fields -> guarded before any HTTP call (no respx route registered).
    assert update_project("p1").startswith("error:")


@respx.mock
def test_delete_project_calls_endpoint() -> None:
    route = respx.delete(f"{_BASE}/api/projects/p1").mock(
        return_value=httpx.Response(200, json={"deleted": True, "current": None})
    )
    out = delete_project("p1")
    assert route.called
    assert "deleted" in out


@respx.mock
def test_update_agent_patches_only_provided_fields() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "ag1", "permission_mode": "auto"})

    respx.patch(f"{_BASE}/api/agents/ag1").mock(side_effect=handler)
    out = update_agent("ag1", permission_mode="auto", backend="claude")
    assert seen["body"] == {"permission_mode": "auto", "backend": "claude"}
    assert "auto" in out


def test_update_agent_requires_a_field() -> None:
    assert update_agent("ag1").startswith("error:")


@respx.mock
def test_delete_agent_calls_endpoint() -> None:
    route = respx.delete(f"{_BASE}/api/agents/ag1").mock(
        return_value=httpx.Response(200, json={"deleted": True, "current": None})
    )
    out = delete_agent("ag1")
    assert route.called
    assert "deleted" in out


def test_agent_write_tools_reject_orchestrator() -> None:
    # The reserved orchestrator edits/removes itself only via settings, so these
    # tools refuse its id BEFORE any HTTP call (no respx route -> a call would fail).
    from scufris.agent_store import ORCHESTRATOR_ID

    assert update_agent(ORCHESTRATOR_ID, permission_mode="auto").startswith("error:")
    assert delete_agent(ORCHESTRATOR_ID).startswith("error:")


@respx.mock
def test_crud_tool_error_path() -> None:
    respx.patch(f"{_BASE}/api/agents/ghost").mock(
        return_value=httpx.Response(404, text="no such agent")
    )
    out = update_agent("ghost", model="gpt-5.5")
    assert out.startswith("error:") and "404" in out and "no such agent" in out


def test_crud_tool_rejects_bad_id() -> None:
    assert get_project("a/b").startswith("error:")
    assert update_project("a b", name="x").startswith("error:")
    assert delete_project("p/../x").startswith("error:")
    assert delete_agent("a/b").startswith("error:")
