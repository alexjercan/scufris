"""Tests for the Scufris MCP tool server.

Tools are called directly (FastMCP's decorator returns the original function).
`host_stats` runs the real collector; the control tools call a respx-stubbed local
dashboard API, so the HTTP plumbing is exercised without a live server or the LLM.
"""

from __future__ import annotations

import json
import shutil
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
    journal_add_macros,
    journal_add_note,
    journal_add_task,
    journal_complete_task,
    journal_log_weight,
    journal_notes,
    journal_remove_task,
    journal_show,
    journal_toggle_habit,
    list_processes,
    list_projects,
    macros_add_food,
    macros_lookup,
    macros_search,
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
        # the-den journal tools (orchestrator-only; den injected via SCUFRIS_DEN_PATH)
        "journal_show",
        "journal_notes",
        "journal_add_task",
        "journal_complete_task",
        "journal_remove_task",
        "journal_toggle_habit",
        "journal_log_weight",
        "journal_add_macros",
        "journal_add_note",
        # macros food-lookup tools (orchestrator-only; wrap the `macros` CLI)
        "macros_lookup",
        "macros_search",
        "macros_add_food",
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
def test_message_agent_forwards_parent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """message_agent stamps the spawn with the orchestrator chat that called it
    (SCUFRIS_ORCH_SESSION_ID), so a later request_input routes back to it (part 3);
    run_agent does the same on its POST body."""
    monkeypatch.setenv("SCUFRIS_ORCH_SESSION_ID", "chat-1")
    seen: dict[str, Any] = {}

    def chat_handler(request: httpx.Request) -> httpx.Response:
        seen["chat"] = json.loads(request.content)
        return httpx.Response(
            200, text='id: 1\ndata: {"kind":"done","reply":{"text":"ok"}}\n\n'
        )

    def run_handler(request: httpx.Request) -> httpx.Response:
        seen["run"] = json.loads(request.content)
        return httpx.Response(200, json={"agent_id": "ag1", "state": "queued"})

    respx.post(f"{_BASE}/api/agents/ag1/chat").mock(side_effect=chat_handler)
    respx.post(f"{_BASE}/api/agents/ag1/run").mock(side_effect=run_handler)

    message_agent("ag1", "hi")
    run_agent("ag1", "do it")
    assert seen["chat"]["parent_session_id"] == "chat-1"
    assert seen["run"]["parent_session_id"] == "chat-1"


@respx.mock
def test_message_agent_no_parent_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh orchestrator turn (no SCUFRIS_ORCH_SESSION_ID) sends no
    parent_session_id, so the child stays unattributed - back-compat."""
    monkeypatch.delenv("SCUFRIS_ORCH_SESSION_ID", raising=False)
    seen: dict[str, Any] = {}
    respx.post(f"{_BASE}/api/agents/ag1/chat").mock(
        side_effect=lambda r: (
            seen.update(body=json.loads(r.content))
            or httpx.Response(
                200, text='id: 1\ndata: {"kind":"done","reply":{"text":"ok"}}\n\n'
            )
        )
    )
    message_agent("ag1", "hi")
    assert "parent_session_id" not in seen["body"]


@respx.mock
def test_pending_agents_scopes_to_the_calling_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pending_agents passes the orchestrator's current chat so the endpoint scopes
    to it (part 3)."""
    monkeypatch.setenv("SCUFRIS_ORCH_SESSION_ID", "chat-1")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=[])

    respx.get(url__startswith=f"{_BASE}/api/agents/pending").mock(side_effect=handler)
    pending_agents()
    assert "parent_session_id=chat-1" in seen["url"]


@respx.mock
def test_pending_agents_renders_parent_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rendered table surfaces each child's parent chat so the operator sees
    the attribution the routing is based on (part 3 review R1.1)."""
    monkeypatch.delenv("SCUFRIS_ORCH_SESSION_ID", raising=False)
    respx.get(url__startswith=f"{_BASE}/api/agents/pending").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "agent_id": "builder",
                    "state": "waiting",
                    "message": "merge?",
                    "run_id": "r1",
                    "session_id": "s1",
                    "ts": 1.0,
                    "parent_agent_id": "orchestrator",
                    "parent_session_id": "chat-1",
                },
                {
                    "agent_id": "loner",
                    "state": "error",
                    "message": "boom",
                    "run_id": "r2",
                    "session_id": "s2",
                    "ts": 2.0,
                    "parent_agent_id": None,
                    "parent_session_id": None,
                },
            ],
        )
    )
    out = pending_agents()
    assert "PARENT" in out
    assert "chat-1" in out  # attributed child shows its chat
    # The unattributed row renders "-" for its parent.
    loner_line = next(line for line in out.splitlines() if line.startswith("loner"))
    assert "-" in loner_line


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


# --- the-den journal tools ----------------------------------------------------
#
# Two layers: (1) gating + argv-construction tests that need NO `today` binary
# (they either short-circuit on a bad den, or capture the argv `_journal` would
# run by stubbing `_run`), so they pin the exact CLI contract deterministically
# and stay green in the pure `nix flake check` sandbox; (2) real end-to-end tests
# that drive the ACTUAL `today` CLI against a temp den, skipped where `today` is
# not on PATH (the sandbox) and run wherever it is installed (a dev box). The
# den is injected as SCUFRIS_DEN_PATH, exactly as agent.scufris_mcp_server does.

_HAS_TODAY = shutil.which("today") is not None
requires_today = pytest.mark.skipif(
    not _HAS_TODAY, reason="the `today` CLI is not on PATH (journal end-to-end tests)"
)

# The daily template `today` renders a new entry from; carrying the real den's
# Habits/Macros/Notes sections so a fresh temp den has habits to toggle.
_DAILY_TEMPLATE = """# {{title}}

### 🌱 Habits

- [ ] 📕 Learn
- [ ] 💪 Gym
- [ ] 🥕 Track Macros

### 🍽️ Macros

what,protein,carbs,fat

### 📝 Notes
"""


@pytest.fixture
def den(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp den (with the daily template) wired as SCUFRIS_DEN_PATH."""
    root = tmp_path / "the-den"
    (root / "Templates").mkdir(parents=True)
    (root / "Templates" / "daily.md").write_text(_DAILY_TEMPLATE)
    monkeypatch.setenv("SCUFRIS_DEN_PATH", str(root))
    return root


# --- gating: no den configured / den missing (no `today` needed) --------------


def _record_run(sink: list[list[str]], ret: str):
    """A `_run` stand-in that records the argv it was handed and returns ``ret`` -
    so a test can assert the tool did (or did not) shell out, and with what."""

    def _fake(args: list[str], **_kw: object) -> str:
        sink.append(args)
        return ret

    return _fake


def test_journal_unconfigured_reports_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """With SCUFRIS_DEN_PATH unset the tools are inert: a clear message and no
    shell-out (so scufris is safe on a box without the-den)."""
    monkeypatch.delenv("SCUFRIS_DEN_PATH", raising=False)
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(called, ""))
    out = journal_show()
    assert out.startswith("error:") and "not configured" in out
    assert called == []  # never shelled out


def test_journal_missing_den_reports_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured-but-absent den short-circuits with a clean error instead of
    letting the raw CLI raise a traceback."""
    monkeypatch.setenv("SCUFRIS_DEN_PATH", "/no/such/den/xyz")
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(called, ""))
    out = journal_add_task("buy milk")
    assert out.startswith("error:") and "does not exist" in out
    assert called == []


def test_journal_expands_tilde_in_den(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `~`-prefixed SCUFRIS_DEN_PATH is expanded at use time (repo convention:
    pydantic stores env Paths verbatim, consumers expanduser), so the
    `~/personal/the-den` form documented in .env.example actually works - both the
    dir check and the `--den` arg see the resolved absolute path, never a `~`."""
    (tmp_path / "den").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCUFRIS_DEN_PATH", "~/den")
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(seen, "{}"))
    journal_show()
    assert seen and seen[0][2] == str(tmp_path / "den")  # --den arg, expanded
    assert "~" not in seen[0][2]


# --- argv construction (no `today` needed: `_run` is stubbed) -----------------


@pytest.fixture
def capture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Point SCUFRIS_DEN_PATH at a real (empty) dir so the den check passes, then
    capture the argv each tool hands to `_run` without invoking `today`."""
    monkeypatch.setenv("SCUFRIS_DEN_PATH", str(tmp_path))
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(seen, "{}"))
    return seen


def test_journal_argv_contract(capture_run: list[list[str]]) -> None:
    """Every tool builds the exact `today --den <den> ...` argv, with global flags
    (--den, -N) BEFORE the subcommand as the CLI requires."""
    journal_show()
    journal_show(offset=-1)
    journal_notes()
    journal_notes(tag="mood")
    journal_add_task("buy milk")
    journal_add_task("call bob", tomorrow=True)
    journal_complete_task(2)
    journal_remove_task(1)
    journal_remove_task(3, tomorrow=True)
    journal_toggle_habit("Gym")
    journal_log_weight("80kg")
    journal_add_macros("eggs,20,2,15")
    journal_add_note("felt great")
    journal_add_note("tagged", tag="mood")
    # Strip the leading `today --den <den>` prefix each call shares.
    tails = [args[3:] for args in capture_run]
    assert all(args[0] == "today" and args[1] == "--den" for args in capture_run)
    assert tails == [
        ["-N", "0", "show", "--json"],
        ["-N", "-1", "show", "--json"],
        ["note", "list", "--json"],
        ["note", "list", "--tag", "mood", "--json"],
        ["task", "add", "buy milk", "--json"],
        ["task", "add", "call bob", "--tomorrow", "--json"],
        ["task", "done", "2", "--json"],
        ["task", "rm", "1", "--json"],
        ["task", "rm", "3", "--tomorrow", "--json"],
        ["habit", "toggle", "Gym", "--json"],
        ["weight", "80kg"],
        ["macros", "add", "eggs,20,2,15", "--json"],
        ["note", "add", "felt great", "--json"],
        ["note", "add", "tagged", "--tag", "mood", "--json"],
    ]


def test_journal_input_guards(capture_run: list[list[str]]) -> None:
    """Empty required text is rejected before shelling out."""
    assert journal_add_task("  ").startswith("error:")
    assert journal_toggle_habit("").startswith("error:")
    assert journal_log_weight("").startswith("error:")
    assert journal_add_macros("").startswith("error:")
    assert journal_add_note("").startswith("error:")
    assert capture_run == []  # none reached `_run`


# --- real `today` CLI against a temp den --------------------------------------


@requires_today
def test_journal_show_reads_the_day(den: Path) -> None:
    data = json.loads(journal_show())
    assert data["date"]  # the CLI stamped a dated entry
    names = {h["name"] for h in data["habits"]}
    assert any("Gym" in n for n in names)
    assert data["tasks"] == [] and data["tomorrow"] == []


@requires_today
def test_journal_task_lifecycle(den: Path) -> None:
    added = json.loads(journal_add_task("buy milk"))
    assert added == [{"index": 1, "text": "buy milk", "done": False}]
    done = json.loads(journal_complete_task(1))
    assert done[0]["done"] is True
    # The mutation is durable: a fresh show reflects it.
    assert json.loads(journal_show())["tasks"][0]["done"] is True
    assert json.loads(journal_remove_task(1)) == []


@requires_today
def test_journal_tomorrow_task(den: Path) -> None:
    added = json.loads(journal_add_task("call bob", tomorrow=True))
    assert added == [{"index": 1, "text": "call bob"}]
    assert json.loads(journal_show())["tomorrow"][0]["text"] == "call bob"
    assert json.loads(journal_remove_task(1, tomorrow=True)) == []


@requires_today
def test_journal_toggle_habit(den: Path) -> None:
    habits = json.loads(journal_toggle_habit("Gym"))
    gym = next(h for h in habits if "Gym" in h["name"])
    assert gym["done"] is True


@requires_today
def test_journal_log_weight_then_show(den: Path) -> None:
    out = journal_log_weight("80kg")
    assert "80" in out
    assert json.loads(journal_show())["weight"] == 80.0


@requires_today
def test_journal_add_macros(den: Path) -> None:
    agg = json.loads(journal_add_macros("eggs,20,2,15"))
    assert agg["protein"] == 20.0 and agg["carbs"] == 2.0 and agg["fat"] == 15.0


@requires_today
def test_journal_notes_add_and_filter(den: Path) -> None:
    journal_add_note("felt great", tag="mood")
    journal_add_note("bought a book")
    assert len(json.loads(journal_notes())) == 2
    mood = json.loads(journal_notes(tag="mood"))
    assert mood == [{"text": "felt great", "tag": "mood"}]


@requires_today
def test_journal_bad_index_is_clean_error(den: Path) -> None:
    """A bad task index surfaces the CLI's one-line stderr, not a traceback."""
    out = journal_complete_task(99)
    assert "no today task" in out.lower() or "error" in out.lower()
    assert "Traceback" not in out


# --- macros food-lookup tools -------------------------------------------------
#
# Same two layers as the journal tools: argv-construction + input-guard tests that
# stub `_run` (no `macros` binary needed, so green in the pure sandbox), plus real
# end-to-end tests that drive the actual `macros` CLI, skipped where it is absent.

_HAS_MACROS = shutil.which("macros") is not None
requires_macros = pytest.mark.skipif(
    not _HAS_MACROS, reason="the `macros` CLI is not on PATH (macros end-to-end tests)"
)


def test_macros_argv_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each tool builds the exact `macros` argv: a bare query for lookup, `-q` for
    search, `-i` for the write."""
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(seen, "ok"))
    macros_lookup("egg 2p")
    macros_search("chick")
    macros_add_food("banana 100g,1,23,0.3")
    assert seen == [
        ["macros", "egg 2p"],
        ["macros", "-q", "chick"],
        ["macros", "-i", "banana 100g,1,23,0.3"],
    ]


def test_macros_input_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty input is rejected before shelling out."""
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.mcp_server._run", _record_run(called, "ok"))
    assert macros_lookup("  ").startswith("error:")
    assert macros_search("").startswith("error:")
    assert macros_add_food("   ").startswith("error:")
    assert called == []


@pytest.fixture
def macros_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp HOME with a seeded macros DB so the real-CLI tests are hermetic (they
    do not read/write the operator's live ~/.local/share/nvim/macros.csv). The
    `macros` CLI resolves its DB from $HOME, and `_run` runs it with the inherited
    env, so monkeypatching HOME redirects it to this temp DB."""
    db = tmp_path / ".local" / "share" / "nvim" / "macros.csv"
    db.parent.mkdir(parents=True)
    db.write_text("egg 1pc,6,0,5\nbanana 100g,1,23,0.3\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    return db


@requires_macros
def test_macros_lookup_returns_csv_row(macros_home: Path) -> None:
    """A real lookup returns the `<food> <amount><unit>,<protein>,<carbs>,<fat>`
    row - the exact shape journal_add_macros consumes - scaled to the amount."""
    out = macros_lookup("egg 2p").strip()
    assert out == "egg 2pc,12,0,10"  # 1pc (6,0,5) scaled to 2pc


@requires_macros
def test_macros_search_lists_matches(macros_home: Path) -> None:
    matches = macros_search("ban")
    assert "banana" in matches.lower()


@requires_macros
def test_macros_add_food_then_lookup_finds_it(macros_home: Path) -> None:
    """add_food writes to the DB (a real insert into the temp copy), so a later
    lookup resolves the new food."""
    added = macros_add_food("oats 40g,5,27,3")
    assert "oats" in added.lower()
    assert macros_lookup("oats 40g").strip() == "oats 40g,5,27,3"


@requires_macros
def test_macros_lookup_unknown_food_is_clean_error(macros_home: Path) -> None:
    """An unknown food surfaces the CLI's message, not a traceback."""
    out = macros_lookup("zzznotarealfood 1p")
    assert "unknown food" in out.lower() or "error" in out.lower()
    assert "Traceback" not in out
