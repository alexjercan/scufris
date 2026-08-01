"""Tests for the Scufris MCP tool server.

Tools are called directly (FastMCP's decorator returns the original function).
`host_stats` runs the real collector; the control tools call a respx-stubbed local
dashboard API, so the HTTP plumbing is exercised without a live server or the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from scufris.agent_store import AgentStore
from scufris.config import Settings
from scufris.enums import AgentState
from scufris.mcp_common import apply_disabled_tools
from scufris.mcp_server import (
    _agent_status_text,
    _list_agents_text,
    acknowledge,
    cancel_agent,
    create_agent,
    create_project,
    delete_agent,
    delete_project,
    get_project,
    list_projects,
    mcp,
    message_agent,
    pending_agents,
    run_agent,
    update_agent,
    update_project,
)
from scufris.projects import ProjectStore


def test_main_configures_logging_and_runs(
    monkeypatch: pytest.MonkeyPatch, restore_tool_registry
) -> None:
    # main() applies the operator disabled-tool set to the live registry before
    # serving, so it must be restored or it could leak a trimmed set into later
    # tests (here nothing is disabled, so it is a no-op).
    ran: list[bool] = []
    monkeypatch.setattr(mcp, "run", lambda: ran.append(True))
    from scufris.mcp_server import main as mcp_main

    mcp_main()
    assert ran == [True]

    # main() also brings the state database to head: this subprocess opens the
    # same file the dashboard does, and nothing guarantees the dashboard ran
    # first. Asserted at the CALL SITE - a test that drove the migration helper
    # directly would still pass with this line deleted from main().
    from scufris.config import Settings
    from scufris.db import open_database
    from scufris.db.migrate import current_revision, head_revision

    db = open_database(Settings().state_dir)
    try:
        assert current_revision(db) == head_revision()
    finally:
        db.close()


async def test_tools_registered() -> None:
    # The scufris server holds ONLY the orchestrator agentic surface now; the life
    # tools live on the `den` server and the callbacks on the `agent` server.
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {
        "host_stats",
        "disk_usage",
        "list_processes",
        # deep read-only host inspection (task 20260729-125024)
        "host_units",
        "host_failed_units",
        "host_unit_status",
        "host_journal",
        "host_storage",
        "host_largest_directories",
        "host_reclaimable_space",
        "host_network",
        "host_thermal",
        "host_what_provides",
        "host_generation_diff",
        "host_flake_status",
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
        "cancel_agent",
        # orchestrator-side agent-comms tools (BC3)
        "pending_agents",
        "acknowledge",
        # NOTE what is NOT here: propose_host_action, propose_nixos_change,
        # host_action_status, nixos_change_status, host_action_audit. The mutating
        # half of the host toolset moved to the host agent's `host` server
        # (tasks/20260729-125040/DECISION.md section 2), so the orchestrator can
        # read this machine but must delegate a CHANGE to it.
    }
    assert all(tool.description for tool in await mcp.list_tools())


async def test_servers_expose_disjoint_tool_sets() -> None:
    """The four servers split the surface by audience, and the ONE overlap is
    deliberate: read-only host inspection is on both `scufris` and `host`, because
    the orchestrator answering "why is this box hot" directly is the point of
    keeping it (tasks/20260729-125040/DECISION.md section 2). Everything else is
    disjoint - and specifically, the MUTATING host tools exist on `host` alone.
    """
    from scufris import (
        agent_mcp_server,
        den_mcp_server,
        host_mcp_server,
        mcp_server,
    )
    from scufris.mcp_host_tools import ACTIONS, INSPECTION

    async def names(m: object) -> set[str]:
        return {t.name for t in await m.mcp.list_tools()}  # type: ignore[attr-defined]

    scufris = await names(mcp_server)
    den = await names(den_mcp_server)
    agent = await names(agent_mcp_server)
    host = await names(host_mcp_server)
    assert len(scufris) == 30 and len(den) == 12 and len(agent) == 2
    assert len(host) == len(INSPECTION) + len(ACTIONS) == 20
    assert scufris.isdisjoint(den)
    assert scufris.isdisjoint(agent)
    assert den.isdisjoint(agent)
    assert den.isdisjoint(host)
    assert host.isdisjoint(agent)
    # The one overlap, stated exactly: inspection, and nothing else.
    inspection = {fn.__name__ for fn in INSPECTION}
    assert scufris & host == inspection
    # The mutating half is the host agent's alone, and the orchestrator has none
    # of it - the guarantee is "not registered", not a runtime filter.
    actions = {fn.__name__ for fn in ACTIONS}
    assert actions <= host
    assert actions.isdisjoint(scufris)
    # The moved tools are truly gone from scufris.
    assert {"journal_show", "macros_lookup", "request_input", "report_back"}.isdisjoint(
        scufris
    )


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
    removed = apply_disabled_tools(mcp, ["disk_usage", "does_not_exist"])
    assert removed == ["disk_usage"]  # only the real one reported
    # The disabled tool is gone from the live registry, so codex never sees it.
    assert mcp._tool_manager.get_tool("disk_usage") is None
    names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "disk_usage" not in names
    assert "host_stats" in names  # others untouched


def test_apply_disabled_tools_empty_is_noop(restore_tool_registry) -> None:
    before = {t.name for t in mcp._tool_manager.list_tools()}
    assert apply_disabled_tools(mcp, []) == []
    after = {t.name for t in mcp._tool_manager.list_tools()}
    assert before == after


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


def test_agent_status_text_surfaces_error_detail(tmp_path: Path) -> None:
    """A run that ended in a backend StreamError persists an ERROR outcome whose
    message is the diagnostic detail; agent_status must surface WHY, not leave
    'state: error' with no reason (cross-process: it re-reads the outcome store)."""
    settings, store = _seed_agent(tmp_path)
    store.mark_finished(
        "builder",
        state=AgentState.ERROR,
        session_id="mock-session",
        message="app-server timed out after 120s",
    )
    out = _agent_status_text(settings, "builder")
    assert "state: error" in out
    assert "error: app-server timed out after 120s" in out


def test_agent_status_text_no_error_line_on_clean_run(tmp_path: Path) -> None:
    """A clean DONE outcome carries no error line - the error surface is reserved
    for a genuinely failed run, so a healthy agent's status stays uncluttered."""
    settings, store = _seed_agent(tmp_path)
    store.mark_finished(
        "builder", state=AgentState.DONE, session_id="mock-session", message="all done"
    )
    out = _agent_status_text(settings, "builder")
    assert "state: done" in out
    assert "error:" not in out


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
def test_cancel_agent_posts_cancel() -> None:
    route = respx.post(f"{_BASE}/api/agents/ag1/cancel").mock(
        return_value=httpx.Response(200, json={"agent_id": "ag1", "cancelled": True})
    )
    out = cancel_agent("ag1")
    assert route.called
    assert "cancelled" in out.lower()


@respx.mock
def test_cancel_agent_reports_no_active_run() -> None:
    respx.post(f"{_BASE}/api/agents/ag1/cancel").mock(
        return_value=httpx.Response(
            404, json={"detail": "no active run for this agent"}
        )
    )
    out = cancel_agent("ag1")
    assert out.startswith("error:")
    assert "no active run" in out


def test_cancel_agent_refuses_orchestrator() -> None:
    # The guard runs before any HTTP call, so no respx route is needed - the
    # orchestrator cannot cancel its own in-flight run from within it.
    out = cancel_agent("orchestrator")
    assert out.startswith("error:")
    assert "orchestrator" in out.lower()


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


def test_agent_write_tools_reject_the_reserved_agents() -> None:
    # Both reserved agents edit/remove themselves only via settings, so these tools
    # refuse their ids BEFORE any HTTP call (no respx route -> a call would fail).
    from scufris.agent_store import RESERVED_AGENT_IDS

    assert RESERVED_AGENT_IDS == {"orchestrator", "host"}
    for agent_id in RESERVED_AGENT_IDS:
        assert update_agent(agent_id, permission_mode="auto").startswith("error:")
        assert delete_agent(agent_id).startswith("error:")


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
