"""Scufris MCP server: a curated set of tools the agent can call.

Exposed over stdio (MCP) and registered with Codex per-invocation by the agent
(no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.

The server is ROLE-SCOPED (agent._mcp_overrides picks the audience via
SCUFRIS_AGENT_ROLE; see apply_role). The ORCHESTRATOR role gets the full surface:
read-only host introspection (host_stats, disk_usage, list_processes), read-only
agent observation (list_agents, agent_status), and orchestrator CONTROL tools that
call the dashboard's own HTTP API - full CRUD over projects (list/get/create/
update/delete) and agents (create/update/delete + run/message), where the write
tools edit REGULAR agents only (the orchestrator configures itself via settings).
A sub-AGENT role gets ONLY the capability-free callback tools (request_input, the
needs-input signal) - it can signal back but cannot create/run/observe agents or
inspect the host. tatr task management is intentionally NOT here: the orchestrator
runs the `tatr` skill via Bash, so a dedicated MCP wrapper would be redundant.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from .metrics import PsutilCollector
from .processes import ProcessList, PsutilProcessCollector

if TYPE_CHECKING:
    # Imported lazily inside the tool helpers (to keep the MCP server's startup
    # import light); named here only for type checking.
    from .agent_store import AgentStore
    from .config import Settings

logger = logging.getLogger(__name__)

# Cap tool output so a huge result can't blow up the model context.
_MAX_OUTPUT = 20_000
_TIMEOUT_SECONDS = 15.0

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


def _run(args: list[str], *, timeout: float = _TIMEOUT_SECONDS) -> str:
    """Run a curated command safely and return bounded combined output.

    `shell=False` with an explicit argument list; the executable is resolved on
    PATH; failures and timeouts are reported as text rather than raised, so the
    agent gets a usable message.
    """
    exe = shutil.which(args[0])
    if exe is None:
        logger.info("run %s: not found on PATH", args[0])
        return f"error: {args[0]} not found on PATH"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.info("run %s: timed out after %ss", args[0], timeout)
        return f"error: {args[0]} timed out after {timeout}s"
    output = proc.stdout
    if proc.returncode != 0:
        logger.info("run %s: exit=%d", args[0], proc.returncode)
        output = (output + "\n" + proc.stderr).strip() or f"exit {proc.returncode}"
    logger.debug(
        "run %s -> exit=%d bytes=%d in %.2fs",
        " ".join(args),
        proc.returncode,
        len(output),
        time.monotonic() - started,
    )
    return output[:_MAX_OUTPUT]


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

_API_TIMEOUT = 15.0
# A steer/chat turn runs a full agent turn, so it needs a longer bound than a
# create/list call; still capped so an MCP tool call cannot hang unboundedly.
_CHAT_TIMEOUT = 120.0


def _api_base() -> str:
    """Base URL of the dashboard's HTTP API (``SCUFRIS_API_BASE``, injected by the
    dashboard when it spawns this server); defaults to the usual local bind."""
    import os

    return os.environ.get("SCUFRIS_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _api_call(
    method: str, path: str, *, body: object | None = None, timeout: float = _API_TIMEOUT
) -> str:
    """Call the local dashboard API and return bounded text (never raises).

    Failures and non-2xx responses come back as ``error: ...`` text, like ``_run``,
    so the model gets a usable message instead of an exception. Output is truncated
    to ``_MAX_OUTPUT``.
    """
    import httpx

    url = _api_base() + path
    try:
        resp = httpx.request(method, url, json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        logger.info("api %s %s: %s", method, path, exc)
        return f"error: request to {path} failed: {exc}"
    if resp.status_code >= 400:
        detail = resp.text.strip()
        logger.info("api %s %s -> %d", method, path, resp.status_code)
        return f"error: {resp.status_code} from {path}: {detail[:500]}"
    return resp.text[:_MAX_OUTPUT]


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
    body = {"goal": goal} if goal is not None else {}
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
    raw = _api_call(
        "POST",
        f"/api/agents/{aid}/chat",
        body={"message": message},
        timeout=_CHAT_TIMEOUT,
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
    waiting for a decision, or that errored. Poll this to find blocked agents -
    especially at the end of a turn - so a stalled sub-agent does not wait forever.

    Read-only. One row per pending agent: id, state (waiting/error) and its
    question / last message. Answer one by messaging or resuming it
    (`message_agent`), then call `acknowledge(id)` so it stops showing here."""
    text = _api_call("GET", "/api/agents/pending")
    if text.startswith("error:"):
        return text
    try:
        rows = json.loads(text)
    except ValueError:
        return text
    if not rows:
        return "no agents are waiting for you"
    header = f"{'ID':<20} {'STATE':<8} MESSAGE"
    lines = [header]
    for r in rows:
        msg = str(r.get("message", "")).replace("\n", " ")[:120]
        lines.append(
            f"{str(r.get('agent_id', ''))[:20]:<20} {str(r.get('state', ''))[:8]:<8} {msg}"
        )
    return "\n".join(lines)


@mcp.tool()
def acknowledge(agent_id: str) -> str:
    """Mark an agent's pending signal handled, so it stops showing in
    `pending_agents()`. Call after you have answered its `request_input` question
    or dealt with its error. Idempotent - acking an agent with nothing pending is
    a harmless no-op."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    return _api_call("POST", f"/api/agents/{aid}/acknowledge")


def _self_agent_id() -> str:
    """This sub-agent's own id (``SCUFRIS_AGENT_ID``, injected by the dashboard
    when it spawns an agent-role server), so ``request_input`` can address the
    caller back to the API."""
    import os

    return os.environ.get("SCUFRIS_AGENT_ID", "").strip()


@mcp.tool()
def request_input(question: str) -> str:
    """Signal that you are BLOCKED and need a decision from the orchestrator before
    you can continue safely (for example: "should I merge to master?"). Records
    your question and returns immediately - END YOUR TURN right after calling this;
    the orchestrator will answer by resuming your session. Use this instead of
    guessing whenever you need approval or input you cannot obtain yourself."""
    q = question.strip()
    if not q:
        return "error: question is required"
    agent_id = _self_agent_id()
    if not agent_id:
        return "error: request_input is unavailable (no agent id in environment)"
    return _api_call(
        "POST", f"/api/agents/{agent_id}/request_input", body={"question": q}
    )


# --- role-scoped tool exposure ------------------------------------------------
# One scufris server serves two audiences (agent._mcp_overrides picks the role via
# SCUFRIS_AGENT_ROLE): the ORCHESTRATOR gets the full host/observe/control
# surface, a sub-AGENT gets only the capability-free callback tools below. T3
# (20260722-222729) scoped scufris to the orchestrator as a CAPABILITY preference
# (no create/run/observe for sub-agents), not a hard isolation boundary; the role
# allowlist preserves that guarantee while letting sub-agents signal back (BC2,
# DECISION.md in tasks/20260723-094303).

ROLE_ORCHESTRATOR = "orchestrator"
ROLE_AGENT = "agent"
# The ONLY tools a non-orchestrator (sub-)agent may reach: a capability-free
# callback surface - no create/run/observe of agents, no host inspection.
_AGENT_ROLE_TOOLS = {"request_input"}


def _role() -> str:
    """The tool audience for this server instance (``SCUFRIS_AGENT_ROLE``, injected
    by the dashboard): ``orchestrator`` (full surface) or ``agent`` (only the
    sub-agent callback tools). Defaults to orchestrator for back-compat."""
    import os

    role = os.environ.get("SCUFRIS_AGENT_ROLE", ROLE_ORCHESTRATOR).strip().lower()
    return role if role in (ROLE_ORCHESTRATOR, ROLE_AGENT) else ROLE_ORCHESTRATOR


def role_tool_names(names: set[str], role: str) -> set[str]:
    """The subset of ``names`` a server in ``role`` exposes: the agent role keeps
    only ``_AGENT_ROLE_TOOLS``; the orchestrator role keeps everything else
    (dropping the agent-only callbacks). Pure - shared by ``apply_role`` (which
    mutates the live registry before serving) and the dashboard's read-only
    per-agent tools listing (``GET /api/agents/{id}/tools``) so the two never drift
    on what a role actually exposes."""
    if role == ROLE_AGENT:
        return names & _AGENT_ROLE_TOOLS
    return names - _AGENT_ROLE_TOOLS


def apply_role(role: str) -> list[str]:
    """Remove every tool not in ``role``'s audience; return those removed. The
    agent role keeps only ``_AGENT_ROLE_TOOLS``; the orchestrator role keeps
    everything else (dropping the agent-only tools). Done before the server serves,
    so a tool outside the role is never advertised - the guard is the server not
    exposing it, not a UI flag."""
    names = {tool.name for tool in mcp._tool_manager.list_tools()}
    keep = role_tool_names(names, role)
    removed: list[str] = []
    for name in sorted(names - keep):
        if mcp._tool_manager.get_tool(name) is not None:
            mcp._tool_manager.remove_tool(name)
            removed.append(name)
    return removed


def _disabled_tools() -> list[str]:
    """Tool names the operator has disabled, from ``SCUFRIS_DISABLED_TOOLS``.

    The dashboard injects this env (comma-separated) when it spawns the server,
    from the runtime-editable ``disabled_tools`` setting.
    """
    import os

    raw = os.environ.get("SCUFRIS_DISABLED_TOOLS", "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def apply_disabled_tools(names: list[str]) -> list[str]:
    """Remove ``names`` from the live tool registry; return those actually removed.

    Done before the server serves any request, so a disabled tool is never
    advertised or callable - enforcement lives here, not in the UI.
    """
    removed: list[str] = []
    for name in names:
        if mcp._tool_manager.get_tool(name) is not None:
            mcp._tool_manager.remove_tool(name)
            removed.append(name)
    return removed


def main() -> None:
    """Run the MCP server over stdio (spawned by Codex).

    This is a separate process from the dashboard, so it configures its own
    logging from ``SCUFRIS_LOG_LEVEL`` (to stderr; codex captures it).
    """
    import os

    from .logsetup import configure_logging

    configure_logging(os.environ.get("SCUFRIS_LOG_LEVEL", "INFO"))
    role = _role()
    removed_role = apply_role(role)
    if removed_role:
        logger.info("role %s: removed %s", role, ", ".join(removed_role))
    removed = apply_disabled_tools(_disabled_tools())
    if removed:
        logger.info("disabled tools: %s", ", ".join(removed))
    mcp.run()


if __name__ == "__main__":
    main()
