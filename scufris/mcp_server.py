"""Scufris `scufris` MCP server: the orchestrator's agentic tools.

Exposed over stdio (MCP) and registered with a backend per-invocation by the
agent (no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.

This is ONE of four single-audience scufris MCP servers:
``scufris`` (this module, orchestrator agentic), ``den`` (``den_mcp_server``, the
operator's journal + macros life tools), ``host`` (``host_mcp_server``, the host
agent's toolset) and ``agent`` (``agent_mcp_server``, the sub-agent callback
tools). Only an ORCHESTRATOR turn registers this server, so its tools are never
advertised to a sub-agent (the guarantee is "not registered", not a runtime
filter).

The surface here: read-only host INSPECTION, registered from
``mcp_host_tools`` (the light live snapshot host_stats / disk_usage /
list_processes plus the deep inspection over ``scufris.host``) - but NOT the
mutating propose tools, which are the host agent's and reach this process only
through delegation, so the propose -> preview -> approve contract is stated to
exactly one audience - read-only agent observation (list_agents, agent_status), the
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
    apply_disabled_tools,
)
from .mcp_host_tools import register as register_host_tools

if TYPE_CHECKING:
    # Imported lazily inside the tool helpers (to keep the MCP server's startup
    # import light); named here only for type checking.
    from .agent_store import AgentStore
    from .config import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("scufris")

# The read-only host INSPECTION half of the host toolset, defined in
# `mcp_host_tools` and registered here. The mutating propose tools are NOT on
# this server: they live on the host agent's `host` server, so the
# propose -> preview -> approve contract is stated to one audience. The
# orchestrator keeps inspection because "why is this box hot" should not cost
# a delegation.
register_host_tools(mcp, actions=False)


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
    # The list always contains at least the reserved HOST agent (the orchestrator
    # is the hidden default and is not listed), so there is no empty case.
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
    from .agent_store import AgentNotFound, AgentState
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
    # A run that ended in error carries WHY on its durable outcome message (a
    # backend StreamError's detail: idle timeout, over-limit line, thread-setup
    # failure). read_status below only reports session progress, never this, so
    # surface the outcome's error explicitly instead of leaving "state: error"
    # with no reason. The outcome is the cross-process substitute for the closed
    # run bus, so this works from the MCP subprocess.
    outcome = store.outcome(agent_id)
    if outcome is not None and outcome.state == AgentState.ERROR and outcome.message:
        # Flatten + cap like the pending_agents row (mcp_server.py pending loop):
        # a StreamError detail can be long or multi-line, and this line sits in a
        # single-line-per-field status block.
        detail = outcome.message.replace("\n", " ")[:200]
        lines.append(f"error: {detail}")
    if outcome is not None and outcome.state == AgentState.CANCELLED:
        # A user stop is a neutral terminal state (not an error); surface it so
        # the orchestrator does not re-read a stale prior message as if live.
        lines.append("cancelled: the run was stopped")
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


def _reject_reserved(agent_id: str) -> str | None:
    """An ``error:`` message if ``agent_id`` is a RESERVED agent, else None.

    The orchestrator and the host agent are both synthetic: they configure
    themselves through the settings store, not these tools, and neither can be
    deleted. These write tools edit REGULAR agents only. The API refuses them
    anyway (409/403); refusing here means the model gets a sentence it can act on
    instead of a status code (review round 1, R1.5)."""
    from .agent_store import RESERVED_AGENT_IDS

    if agent_id in RESERVED_AGENT_IDS:
        return (
            f"error: {agent_id!r} is a reserved agent - it configures itself via "
            "settings and cannot be edited or deleted with this tool"
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
def cancel_agent(agent_id: str) -> str:
    """Cancel a sub-agent's in-flight run (stop what it is currently doing).

    Use this when asked to "cancel that agent" / "stop agent-<id>". It truly
    aborts the running turn (the backend process is stopped), and the agent's
    terminal state becomes "cancelled". 404 unknown agent or no active run (there
    is nothing to cancel). You cannot cancel the orchestrator's OWN run from here
    - the user stops that with the chat stop button."""
    from .agent_store import ORCHESTRATOR_ID

    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    if aid == ORCHESTRATOR_ID:
        return (
            "error: cannot cancel your own (orchestrator) run from within it; "
            "the user stops it with the chat stop button"
        )
    return _api_call("POST", f"/api/agents/{aid}/cancel")


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
    codex or claude (the provider); ``model`` is the LLM. The reserved agents (the
    orchestrator and the host agent) configure themselves via settings and cannot be
    edited here. Returns the updated
    agent. 404 unknown, 422 invalid field, 403 read-only."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_reserved(aid)
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

    The reserved agents (the orchestrator, the host agent) cannot be deleted. 404
    unknown, 403 read-only or reserved."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_reserved(aid)
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
