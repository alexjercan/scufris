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
    _format_processes,
    _list_agents_text,
    acknowledge,
    cancel_agent,
    create_agent,
    create_project,
    delete_agent,
    delete_project,
    disk_usage,
    get_project,
    host_failed_units,
    host_flake_status,
    host_generation_diff,
    host_journal,
    host_network,
    host_reclaimable_space,
    host_stats,
    host_storage,
    host_thermal,
    host_unit_status,
    host_units,
    host_what_provides,
    list_processes,
    list_projects,
    mcp,
    message_agent,
    pending_agents,
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
        # host actions: an agent may PROPOSE a privileged change and read what
        # happened to it. There is deliberately no approve tool.
        "propose_host_action",
        "host_action_status",
        "host_action_audit",
    }
    assert all(tool.description for tool in await mcp.list_tools())


async def test_servers_expose_disjoint_tool_sets() -> None:
    # The three servers partition the tools with no overlap: scufris agentic, den
    # life, agent callbacks. This is the split's core guarantee.
    from scufris import agent_mcp_server, den_mcp_server, mcp_server

    async def names(m: object) -> set[str]:
        return {t.name for t in await m.mcp.list_tools()}  # type: ignore[attr-defined]

    scufris = await names(mcp_server)
    den = await names(den_mcp_server)
    agent = await names(agent_mcp_server)
    assert len(scufris) == 33 and len(den) == 12 and len(agent) == 2
    assert scufris.isdisjoint(den)
    assert scufris.isdisjoint(agent)
    assert den.isdisjoint(agent)
    # The moved tools are truly gone from scufris.
    assert {"journal_show", "macros_lookup", "request_input", "report_back"}.isdisjoint(
        scufris
    )


async def test_host_tool_descriptions_steer_away_from_shell() -> None:
    # The tool descriptions are one of the model's signals; they should explicitly
    # tell it to prefer these over raw shell (the real steering is the prompt
    # preamble in agent.py, but strong descriptions reinforce it).
    desc = {tool.name: (tool.description or "") for tool in await mcp.list_tools()}
    assert "PREFERRED" in desc["host_stats"] or "instead of shell" in desc["host_stats"]
    assert "uname" in desc["host_stats"] and "/proc" in desc["host_stats"]
    assert "PREFER" in desc["disk_usage"]
    assert "PREFER" in desc["list_processes"]
    # Every deep-inspection tool carries the same steering, and names the shell
    # command it replaces so the model can match its instinct to a tool.
    replaces = {
        "host_units": "systemctl list-units",
        "host_failed_units": "systemctl --failed",
        "host_unit_status": "systemctl status",
        "host_journal": "journalctl",
        "host_storage": "df",
        "host_largest_directories": "du",
        "host_reclaimable_space": "nix-collect-garbage",
        "host_network": "iptables",
        "host_thermal": "sensors",
        "host_what_provides": "which",
        "host_generation_diff": "nix store diff-closures",
        "host_flake_status": "flake.lock",
    }
    for name, shell in replaces.items():
        assert "PREFER" in desc[name], f"{name} does not steer away from shell"
        assert shell in desc[name], f"{name} does not name the `{shell}` it replaces"
    # The two tools whose cost is the trap say so, so the model does not poll
    # them. Whitespace-normalised: these phrases sit across a docstring wrap.
    flat = {name: " ".join(text.split()) for name, text in desc.items()}
    assert "tens of seconds" in flat["host_largest_directories"]
    assert "take a minute" in flat["host_reclaimable_space"]
    # And the read-only guarantee is stated where it would be tempting to break.
    assert "read-only" in desc["host_reclaimable_space"].lower()


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


# --- host inspection tools (task 20260729-125024) ----------------------------
#
# These run against the REAL host, like `host_stats` above: they are read-only,
# and a fake would prove only that the fake works. The parsers themselves are
# pinned against captured fixtures in `test_host_inspection.py`; what these
# assert is the TOOL contract - a non-empty string, never an exception, and the
# honesty markers actually reaching the text a model would read.


@pytest.mark.parametrize(
    "call",
    [
        lambda: host_units(state="failed"),
        lambda: host_failed_units(),
        lambda: host_failed_units(scope="user"),
        lambda: host_unit_status("sshd.service"),
        lambda: host_journal(lines=3, since="10 min ago"),
        lambda: host_storage(),
        lambda: host_network(),
        lambda: host_thermal(),
        lambda: host_what_provides("sh"),
        lambda: host_flake_status(),
    ],
    ids=[
        "units",
        "failed-system",
        "failed-user",
        "unit-status",
        "journal",
        "storage",
        "network",
        "thermal",
        "what-provides",
        "flake-status",
    ],
)
def test_host_tools_return_text_and_never_raise(call: Any) -> None:
    out = call()
    assert isinstance(out, str)
    assert out.strip(), "a host tool returned nothing at all"
    # Whatever the outcome, the first line names the report - so a model always
    # knows what it asked about, even when the answer is "unavailable".
    assert len(out.splitlines()) >= 2


def test_host_tools_reject_an_unknown_scope() -> None:
    """A wrong scope is refused, not defaulted: a user unit and a system unit can
    share a name, so silently picking one would answer a different question."""
    for out in (
        host_units(scope="nonsense"),
        host_failed_units(scope="nonsense"),
        host_unit_status("sshd.service", scope="nonsense"),
        host_journal(scope="nonsense"),
    ):
        assert out.startswith("error:")
        assert "system" in out and "user" in out


def test_host_journal_rejects_an_unknown_priority() -> None:
    out = host_journal(priority="urgent", lines=1)
    assert "unknown priority" in out


def test_host_network_states_the_privilege_limit_in_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared-vs-live firewall caveat must reach the model, not just the
    model object - this is the text an agent would repeat to the operator.

    Driven through a fixture system tree rather than the live host: an
    `if "unavailable" not in out` guard would make the assertion silently vanish
    on any host whose firewall report degrades, which is precisely the shape a
    test guarding an honesty property must not have.
    """
    from scufris.host import HostInspector

    script = tmp_path / "store" / "abc-firewall-start" / "bin" / "firewall-start"
    script.parent.mkdir(parents=True)
    script.write_text("ip46tables -A nixos-fw -p tcp --dport 22 -j nixos-fw-accept\n")
    unit = tmp_path / "etc" / "systemd" / "system"
    unit.mkdir(parents=True)
    (unit / "firewall.service").write_text(f"ExecStart=@{script} firewall-start\n")
    monkeypatch.setattr(
        "scufris.mcp_server._inspector", lambda: HostInspector(system=tmp_path)
    )

    out = host_network()
    assert "DECLARED" in out
    assert "needs root" in out
    assert "tcp open: 22" in out


def test_host_generation_diff_defaults_to_the_last_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no arguments it compares PREVIOUS -> CURRENT.

    Asserted at the argv, because that is the only place the claim is visible:
    every render path emits a "closure diff ..." title, so a text assertion here
    would be a tautology that passes whichever generations were compared.
    """
    from scufris.host import CommandResult, HostInspector, Outcome

    generations = json.dumps(
        [
            {"generation": 191, "date": "d", "kernelVersion": "k", "current": True},
            {"generation": 190, "date": "d", "kernelVersion": "k", "current": False},
            {"generation": 12, "date": "d", "kernelVersion": "k", "current": False},
        ]
    )
    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        stdout = generations if argv[0] == "nixos-rebuild" else "linux: 1 -> 2"
        return CommandResult(argv=argv, outcome=Outcome.OK, stdout=stdout, returncode=0)

    monkeypatch.setattr("scufris.mcp_server._inspector", lambda: HostInspector(spy))
    out = host_generation_diff()

    diff_argv = [a for a in seen if a[0] == "nix"]
    assert diff_argv, "no closure diff ran"
    joined = " ".join(diff_argv[0])
    # Previous -> current, in that order. Not 12 (the oldest) and not reversed.
    assert "system-190-link" in joined
    assert "system-191-link" in joined
    assert joined.index("system-190-link") < joined.index("system-191-link")
    assert "system-12-link" not in joined
    assert "linux" in out


def test_host_tools_refuse_an_argument_that_would_become_an_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit name/pattern starting with '-' is refused, never passed through.

    `shell=False` stops shell injection but NOT option injection: measured on
    this host, `systemctl ... -Hsomeone@elsewhere` makes systemctl open an
    outbound SSH connection to a caller-chosen host. These arguments can come
    from a model that just read attacker-influenced text, so the refusal is
    asserted at the tool boundary AND the argv is checked to prove nothing ran.
    """
    from scufris.host import CommandResult, HostInspector, Outcome

    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        return CommandResult(argv=argv, outcome=Outcome.OK, stdout="[]", returncode=0)

    monkeypatch.setattr("scufris.mcp_server._inspector", lambda: HostInspector(spy))
    hostile = "-Hattacker@evil.example.com"
    for out in (
        host_units(pattern=hostile),
        host_unit_status(hostile),
    ):
        assert "unavailable" in out
        assert "'-'" in out
    assert not seen, f"a command ran with a hostile argument: {seen}"


def test_host_reclaimable_space_never_collects_for_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-only guarantee at the tool boundary: only an enumerating argv.

    The real command walks the whole store, so the inspector is swapped here -
    what is under test is the argv, not nix.
    """
    from scufris.host import CommandResult, HostInspector, Outcome

    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        return CommandResult(
            argv=argv,
            outcome=Outcome.OK,
            stdout="12 store paths would be deleted",
            returncode=0,
        )

    # Swap the INSPECTOR the tool builds, not the module-level `run_command`:
    # HostInspector binds its default runner at definition time, so patching the
    # function afterwards would leave the real one in place and the spy empty.
    monkeypatch.setattr("scufris.mcp_server._inspector", lambda: HostInspector(spy))
    out = host_reclaimable_space()
    assert seen, "no command ran"
    for argv in seen:
        # --print-dead only ENUMERATES. There is no --delete-older-than here
        # even in dry-run form: that flag also trims profile generations, which
        # would make read-only-ness a property of nix's behaviour rather than
        # of this code.
        assert argv[:3] == ["nix-store", "--gc", "--print-dead"], argv
        assert "--delete-older-than" not in " ".join(argv)
        assert "-d" not in argv
    assert "12 store paths" in out
    assert "not a size" in out


def test_the_agent_has_no_tool_that_approves_a_host_action() -> None:
    """An agent may propose a privileged change. It may never approve one.

    Enforced twice, and this is the cheap half: no tool exists. The other half
    is the middleware refusing the machine bearer token these subprocesses hold
    (tests/test_host_action_api.py). Both, because a tool added for convenience
    would silently undo the expensive one.
    """
    import scufris.mcp_server as server

    names = {
        name
        for name in dir(server)
        if not name.startswith("_") and callable(getattr(server, name))
    }
    approving = {
        name
        for name in names
        if "host" in name and ("approve" in name or "apply" in name)
    }
    assert not approving, f"an agent-facing approval tool exists: {approving}"
    assert "propose_host_action" in names


def _proposal_payload() -> dict[str, Any]:
    """A HostActionRecord as the API returns one, built from the real models."""
    from scufris.host_actions import HostActionRecord
    from scufris.hostd.actions import ActionKind, RiskClass
    from scufris.hostd.preview import PreviewKind
    from scufris.hostd.protocol import Fingerprint, Preview, ProposalView, Reversal

    view = ProposalView(
        id="a" * 32,
        kind=ActionKind.UNIT_RESTART,
        risk=RiskClass.R1,
        args={"unit": "nginx.service"},
        argv=["systemctl", "restart", "--", "nginx.service"],
        summary="restart nginx.service",
        preview=Preview(
            kind=PreviewKind.STATE,
            headline="nginx.service is active (running)",
            label="Current state and blast radius - not a prediction.",
            lines=["ActiveState=active", "2 units depend on it"],
        ),
        reversal=Reversal(possible=True, summary="start it again if it stays down"),
        fingerprint=Fingerprint(value="f1", describes="nginx.service"),
        created_at=1.0,
        expires_at=601.0,
    )
    return HostActionRecord(proposal=view).model_dump(mode="json")


@respx.mock
def test_the_host_action_tool_returns_the_rendered_preview_not_json() -> None:
    """The tool hands the model prose, not a blob to paraphrase.

    Its own instruction is "show the operator the preview verbatim", which is
    only possible if the preview text is what comes back (review round 1, R1.11;
    unpinned until review round 2, R2.4).
    """
    from scufris.mcp_server import propose_host_action

    respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(200, json=_proposal_payload())
    )

    out = propose_host_action("unit_restart", unit="nginx")

    assert not out.lstrip().startswith("{"), f"the tool returned raw JSON: {out[:80]}"
    assert "Current state and blast radius" in out  # the honesty label, verbatim
    assert "you cannot approve" in out
    assert "the operator must" in out


@respx.mock
def test_a_host_action_tool_error_passes_through_unrendered() -> None:
    """An `error: ...` line is a diagnosable answer; do not turn it into a parse
    failure by insisting on JSON."""
    from scufris.mcp_server import propose_host_action

    respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(503, text="no privileged helper configured")
    )

    out = propose_host_action("unit_restart", unit="nginx")
    assert out.startswith("error:")
    assert "no privileged helper configured" in out


@respx.mock
def test_the_host_action_tool_names_the_agent_it_is_running_as(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit's "which agent" field comes from this process's own id.

    The API derives the ACTOR from the credential and will not let a body field
    promote a machine caller (review round 1, R1.6); this half only makes the
    record say something more useful than "an agent".
    """
    from scufris.mcp_server import propose_host_action

    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    route = respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(200, json=_proposal_payload())
    )

    propose_host_action("unit_restart", unit="nginx")

    body = json.loads(route.calls[0].request.content)
    assert body["agent"] == "builder"
    assert body["kind"] == "unit_restart"
    assert body["args"] == {"unit": "nginx"}
