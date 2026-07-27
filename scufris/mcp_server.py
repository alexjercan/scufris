"""Scufris `scufris` MCP server: the orchestrator's agentic tools.

Exposed over stdio (MCP) and registered with a backend per-invocation by the
agent (no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.

This is ONE of three single-audience scufris MCP servers (see
``tasks/20260727-105609/DECISION.md``): ``scufris`` (this module, orchestrator
agentic), ``den`` (``den_mcp_server``, the operator's journal + macros life
tools) and ``agent`` (``agent_mcp_server``, the sub-agent callback tools). Only
an ORCHESTRATOR turn registers this server, so its tools are never advertised to
a sub-agent (the guarantee is "not registered", not a runtime filter).

The surface here: read-only host introspection (host_stats, disk_usage,
list_processes), read-only agent observation (list_agents, agent_status), the
orchestrator CONTROL tools that call the dashboard's own HTTP API - full CRUD
over projects (list/get/create/update/delete) and agents (create/update/delete +
run/message), where the write tools edit REGULAR agents only (the orchestrator
configures itself via settings) - and the escalation-inbox tools (pending_agents,
acknowledge) that surface sub-agents' callbacks. tatr task management is
intentionally NOT here: the orchestrator runs the `tatr` skill via Bash, so a
dedicated MCP wrapper would be redundant.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from .mcp_common import (
    _MAX_OUTPUT,
    _api_call,
    _disabled_tools,
    _run,
    apply_disabled_tools,
)
from .metrics import PsutilCollector
from .processes import ProcessList, PsutilProcessCollector

if TYPE_CHECKING:
    # Imported lazily inside the tool helpers (to keep the MCP server's startup
    # import light); named here only for type checking.
    from .agent_store import AgentStore
    from .config import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("scufris")
_collector = PsutilCollector()
# Primed at import so per-process cpu% is a real delta on the first sample
# (psutil.process_iter reuses Process objects internally).
_proc_collector = PsutilProcessCollector()


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _format_processes(plist: ProcessList, limit: int) -> str:
    """Render the top application groups as a compact fixed-width table."""
    header = f"{'APPLICATION':<24} {'CPU%':>6} {'MEM':>8} {'#':>4}"
    lines = [header, f"total processes: {plist.total}"]
    for group in plist.groups[: max(1, limit)]:
        lines.append(
            f"{group.name[:24]:<24} {group.cpu_percent:>6.1f} "
            f"{_human_bytes(group.mem_rss):>8} {group.count:>4}"
        )
    return "\n".join(lines)


@mcp.tool()
def host_stats() -> dict[str, object]:
    """Complete structured snapshot of THIS host's live metrics as JSON: CPU model
    and per-core load, memory, swap, disks, network throughput, load average,
    uptime, and GPUs.

    This is the PREFERRED, authoritative way to answer any question about the
    host's current state, hardware, or resource usage. Call this FIRST and use its
    result instead of shell commands like `uname`, `lscpu`, `top`, `free`,
    `uptime`, `nvidia-smi`, or reading `/proc` - one call replaces piecing together
    shell output, and it is curated for this exact host.
    """
    return _collector.sample().model_dump(mode="json")


@mcp.tool()
def disk_usage() -> str:
    """Disk usage per real filesystem (like `df -h`), excluding tmpfs/overlay noise.

    PREFER this over running `df` yourself in the shell for any "how full are my
    disks" question - it is already filtered to the real filesystems on this host.
    """
    return _run(
        [
            "df",
            "-h",
            "-x",
            "tmpfs",
            "-x",
            "devtmpfs",
            "-x",
            "squashfs",
            "-x",
            "overlay",
        ]
    )


@mcp.tool()
def list_processes(limit: int = 15) -> str:
    """Top running applications by CPU, grouped by name (like a compact htop).

    PREFER this over shell `ps`/`top`/`htop` for any "what is using CPU/memory"
    question - it is already grouped and ranked for this host.
    """
    return _format_processes(_proc_collector.sample(), limit)


# --- orchestrator observation (read-only) -----------------------------------
#
# These let the MAIN chat agent (the orchestrator) see the agents running on
# projects, so it can answer "what is agent-N working on". They run in THIS MCP
# subprocess, which does not share the dashboard's in-memory Supervisor - so they
# read PERSISTED state: the AgentStore (agents.json, whose lifecycle the run
# engine persists via mark_running/mark_finished) plus the backend's read_status
# from the rollout/session files. Read-only: no launching or steering (v1).


def _agent_store(settings: "Settings") -> "AgentStore":
    from .agent_store import AgentStore
    from .projects import ProjectStore

    return AgentStore(settings, ProjectStore(settings))


def _list_agents_text(settings: "Settings") -> str:
    # The list always contains at least the reserved orchestrator, so there is
    # no empty case.
    agents = _agent_store(settings).list()
    header = f"{'ID':<20} {'STATE':<9} {'BACKEND':<10} {'PROJECT':<16} NAME"
    lines = [header]
    for a in agents:
        lines.append(
            f"{a.id[:20]:<20} {a.state[:9]:<9} {a.backend[:10]:<10} "
            f"{a.project_id[:16]:<16} {a.name}"
        )
    return "\n".join(lines)


def _agent_status_text(settings: "Settings", agent_id: str) -> str:
    from .agent_store import AgentNotFound
    from .backends import get_backend

    store = _agent_store(settings)
    try:
        agent = store.get(agent_id)
    except AgentNotFound:
        return f"error: no such agent: {agent_id}"
    lines = [
        f"agent {agent.id} ({agent.name})",
        f"state: {agent.state}",
        f"backend: {agent.backend}",
        f"project: {agent.project_id}",
        f"goal: {agent.goal or '-'}",
        f"mode: {agent.permission_mode}",
    ]
    try:
        status = get_backend(agent.backend).read_status(settings, agent.session_id)
    except Exception as exc:  # noqa: BLE001 - never fail the tool on a read
        status = None
        lines.append(f"(progress unavailable: {exc})")
    if status is not None:
        lines += [
            f"turns: {status.turns}",
            f"tool calls: {status.tool_calls}",
            f"tokens in/out: {status.input_tokens}/{status.output_tokens}",
            f"last message: {status.last_message or '-'}",
        ]
    return "\n".join(lines)


@mcp.tool()
def list_agents() -> str:
    """List the orchestrator's configured agents and their current state, so you
    can answer "what agents exist and what is each doing".

    Read-only. One row per agent: id, state (idle/running/blocked/done/error),
    backend (codex/claude), project, and name. Use `agent_status(id)` for detail.
    """
    from .config import Settings

    return _list_agents_text(Settings())


@mcp.tool()
def agent_status(agent_id: str) -> str:
    """Report one agent's current state and progress, so you can answer "what is
    agent-<id> working on".

    Read-only. Returns the agent's config (backend, project, goal, write posture),
    its lifecycle state, and - from its session - turns, tool calls, token usage
    and the last message. Returns a clear error if the id is unknown. This
    OBSERVES an agent; it does not launch or steer it.
    """
    from .config import Settings

    return _agent_status_text(Settings(), agent_id)


# --- orchestrator control (write) -------------------------------------------
#
# These let the orchestrator DO the dashboard's control actions - create/run/
# steer agents, create/list projects - by calling the dashboard's OWN HTTP API
# at 127.0.0.1:<port>. The MCP server is a separate process and cannot touch the
# live in-app Supervisor (the read-only observe tools above work around that by
# reading persisted state); a control action that launches or steers a run needs
# the app process that owns the supervisor, so it crosses back over HTTP, reusing
# every endpoint's validation and lifecycle. The base URL is injected as
# ``SCUFRIS_API_BASE`` when the dashboard spawns this (orchestrator-only) server.
# These tools are only registered for the orchestrator (see agent._mcp_overrides),
# so a regular agent can never create agents or projects.


def _clean_id(value: str) -> str | None:
    """An id argument, stripped; None if empty or holding a `/` or whitespace that
    would break the URL segment it is interpolated into (makes the boundary
    explicit instead of relying on the server to 404)."""
    value = value.strip()
    if not value or "/" in value or any(c.isspace() for c in value):
        return None
    return value


def _provided(**fields: object | None) -> dict[str, object]:
    """A PATCH body of only the fields the caller actually set (not None).

    ``ProjectUpdate`` / ``AgentUpdate`` are ``extra="forbid"`` and all-optional, so
    an unset field must be OMITTED, not sent as null. (An explicit empty string is
    kept - that legitimately clears a field.)"""
    return {name: value for name, value in fields.items() if value is not None}


def _reject_orchestrator(agent_id: str) -> str | None:
    """An ``error:`` message if ``agent_id`` is the reserved orchestrator, else None.

    The orchestrator configures itself through the settings store, not these tools,
    and cannot be deleted; these write tools edit REGULAR agents only."""
    from .agent_store import ORCHESTRATOR_ID

    if agent_id == ORCHESTRATOR_ID:
        return (
            f"error: {ORCHESTRATOR_ID!r} configures itself via settings and cannot "
            "be edited or deleted with this tool"
        )
    return None


@mcp.tool()
def list_projects() -> str:
    """List the registered projects (id, name, language, path), so you can pick one
    to create an agent on or answer "what projects exist".

    Read-only. One row per project."""
    text = _api_call("GET", "/api/projects")
    if text.startswith("error:"):
        return text
    try:
        projects = json.loads(text)
    except ValueError:
        return text
    if not projects:
        return "no projects registered"
    header = f"{'ID':<20} {'LANGUAGE':<12} {'NAME':<20} PATH"
    lines = [header]
    for p in projects:
        lines.append(
            f"{str(p.get('id', ''))[:20]:<20} {str(p.get('language', ''))[:12]:<12} "
            f"{str(p.get('name', ''))[:20]:<20} {p.get('cwd', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def create_project(
    name: str, cwd: str, language: str = "", description: str = ""
) -> str:
    """Register an EXISTING directory as a project (so agents can be created on it).

    ``cwd`` is the absolute path of the project directory. Returns the created
    project, or a clear error (422 bad name/path, 409 duplicate, 403 read-only)."""
    name = name.strip()
    cwd = cwd.strip()
    if not name or not cwd:
        return "error: name and cwd are required"
    return _api_call(
        "POST",
        "/api/projects",
        body={
            "name": name,
            "cwd": cwd,
            "language": language,
            "description": description,
        },
    )


@mcp.tool()
def get_project(project_id: str) -> str:
    """Show one project's detail (id, name, language, path, description) by id.

    Read-only. 404 if the id is unknown."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    return _api_call("GET", f"/api/projects/{pid}")


@mcp.tool()
def update_project(
    project_id: str,
    name: str | None = None,
    cwd: str | None = None,
    language: str | None = None,
    description: str | None = None,
) -> str:
    """Edit a registered project - only the fields you pass change.

    Returns the updated project. 404 unknown, 422 invalid, 403 read-only."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    body = _provided(name=name, cwd=cwd, language=language, description=description)
    if not body:
        return "error: nothing to update - pass at least one field to change"
    return _api_call("PATCH", f"/api/projects/{pid}", body=body)


@mcp.tool()
def delete_project(project_id: str) -> str:
    """Delete (unregister) a project by id.

    Removes the project record; it does not delete the directory on disk. 404
    unknown, 403 read-only."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    return _api_call("DELETE", f"/api/projects/{pid}")


@mcp.tool()
def create_agent(
    name: str,
    project_id: str,
    backend: str | None = None,
    model: str | None = None,
    description: str = "",
    goal: str = "",
    permission_mode: str = "manual",
) -> str:
    """Create an agent bound to a project (from ``list_projects``).

    ``backend`` is codex or claude (server default when omitted); ``permission_mode``
    is manual|edit|auto (read-only by default). Returns the created agent, or a clear
    error (422 bad field / unknown project, 403 read-only)."""
    name = name.strip()
    project_id = project_id.strip()
    if not name or not project_id:
        return "error: name and project_id are required"
    body: dict[str, object] = {
        "name": name,
        "project_id": project_id,
        "description": description,
        "goal": goal,
        "permission_mode": permission_mode,
    }
    if backend:
        body["backend"] = backend
    if model:
        body["model"] = model
    return _api_call("POST", "/api/agents", body=body)


@mcp.tool()
def run_agent(agent_id: str, goal: str | None = None) -> str:
    """Launch a supervised background run for an agent, on its project.

    ``goal`` overrides the agent's stored goal for this run (required if the agent
    has none). Returns immediately with the run's state (usually "queued"); use
    ``agent_status`` to follow progress. 404 unknown agent, 422 no goal, 409 a run
    is already active."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    body: dict[str, str] = {}
    if goal is not None:
        body["goal"] = goal
    parent = _orch_session_id()
    if parent:
        body["parent_session_id"] = parent
    return _api_call("POST", f"/api/agents/{aid}/run", body=body)


@mcp.tool()
def message_agent(agent_id: str, message: str) -> str:
    """Send one chat turn to an agent (steer it / ask it something), resuming its
    session, and return its reply.

    This runs a full agent turn, so it can take a while. 404 unknown agent, 422
    empty message, 409 a turn is already active."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    if not message.strip():
        return "error: message must not be empty"
    body: dict[str, str] = {"message": message}
    parent = _orch_session_id()
    if parent:
        body["parent_session_id"] = parent
    raw = _api_call(
        "POST",
        f"/api/agents/{aid}/chat",
        body=body,
        read_unbounded=True,
    )
    if raw.startswith("error:"):
        return raw
    # The chat endpoint streams SSE frames; collect the assistant reply from them.
    reply = ""
    deltas: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue
        kind = event.get("kind")
        if kind == "done":
            reply = (event.get("reply") or {}).get("text", "") or reply
        elif kind == "text_delta":
            deltas.append(event.get("delta", ""))
        elif kind == "error":
            return f"error: {event.get('detail', 'turn failed')}"
    text = reply or "".join(deltas)
    return (text or "(no reply)")[:_MAX_OUTPUT]


@mcp.tool()
def update_agent(
    agent_id: str,
    name: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    description: str | None = None,
    goal: str | None = None,
    permission_mode: str | None = None,
) -> str:
    """Edit a REGULAR agent's config - only the fields you pass change.

    ``permission_mode`` is manual|edit|auto (read-only|edit|full); ``backend`` is
    codex or claude (the provider); ``model`` is the LLM. The reserved orchestrator
    configures itself via settings and cannot be edited here. Returns the updated
    agent. 404 unknown, 422 invalid field, 403 read-only."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_orchestrator(aid)
    if rejected is not None:
        return rejected
    body = _provided(
        name=name,
        backend=backend,
        model=model,
        description=description,
        goal=goal,
        permission_mode=permission_mode,
    )
    if not body:
        return "error: nothing to update - pass at least one field to change"
    return _api_call("PATCH", f"/api/agents/{aid}", body=body)


@mcp.tool()
def delete_agent(agent_id: str) -> str:
    """Delete a REGULAR agent by id.

    The reserved orchestrator cannot be deleted. 404 unknown, 403 read-only or
    reserved."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_orchestrator(aid)
    if rejected is not None:
        return rejected
    return _api_call("DELETE", f"/api/agents/{aid}")


@mcp.tool()
def pending_agents() -> str:
    """List the sub-agents that need YOU: those that called `request_input` and are
    waiting for a decision, those that called `report_back` and have finished, or
    those that errored. Poll this - especially at the end of a turn - so a stalled or
    finished sub-agent does not go unnoticed.

    Read-only. One row per pending agent: id, state (waiting/reported/error) and its
    question / result summary / last message. Scoped to THIS chat: children this chat
    spawned, plus unattributed ones (UI-launched), but not another chat's children
    (part 3). A `waiting` agent you answer by resuming it (`message_agent`); a
    `reported` agent has finished, so just read its report. Then call
    `acknowledge(id)` so it stops showing here."""
    parent = _orch_session_id()
    path = "/api/agents/pending"
    if parent:
        path = f"{path}?parent_session_id={quote(parent, safe='')}"
    text = _api_call("GET", path)
    if text.startswith("error:"):
        return text
    try:
        rows = json.loads(text)
    except ValueError:
        return text
    if not rows:
        return "no agents are waiting for you"
    header = f"{'ID':<20} {'STATE':<8} {'PARENT':<12} MESSAGE"
    lines = [header]
    for r in rows:
        msg = str(r.get("message", "")).replace("\n", " ")[:120]
        # Which chat spawned this child ("-" when unattributed), so the operator
        # sees the attribution the routing is based on (part 3).
        parent_sess = str(r.get("parent_session_id") or "-")[:12]
        lines.append(
            f"{str(r.get('agent_id', ''))[:20]:<20} "
            f"{str(r.get('state', ''))[:8]:<8} "
            f"{parent_sess:<12} {msg}"
        )
    return "\n".join(lines)


@mcp.tool()
def acknowledge(agent_id: str) -> str:
    """Mark an agent's pending signal handled, so it stops showing in
    `pending_agents()`. Call after you have answered its `request_input` question,
    read its `report_back` result, or dealt with its error. Idempotent - acking an
    agent with nothing pending is a harmless no-op."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    return _api_call("POST", f"/api/agents/{aid}/acknowledge")


def _orch_session_id() -> str:
    """The orchestrator's current chat (``SCUFRIS_ORCH_SESSION_ID``, injected on the
    orchestrator MCP server per turn), so ``message_agent`` / ``run_agent`` can
    stamp a spawned child with the chat that launched it and ``pending_agents`` can
    scope to that chat (part 3). Empty on a fresh orchestrator turn."""
    import os

    return os.environ.get("SCUFRIS_ORCH_SESSION_ID", "").strip()


def main() -> None:
    """Run the orchestrator agentic MCP server over stdio (spawned by a backend).

    This is a separate process from the dashboard, so it configures its own
    logging from ``SCUFRIS_LOG_LEVEL`` (to stderr; the backend captures it). Only
    an orchestrator turn registers this server, so there is no role filtering to
    do - the audience split is physical (see the module docstring); the operator
    disabled-tool set is still applied here.
    """
    import os

    from .logsetup import configure_logging

    configure_logging(os.environ.get("SCUFRIS_LOG_LEVEL", "INFO"))
    removed = apply_disabled_tools(mcp, _disabled_tools())
    if removed:
        logger.info("disabled tools: %s", ", ".join(removed))
    mcp.run()


if __name__ == "__main__":
    main()
