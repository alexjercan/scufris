"""Scufris MCP server: a curated set of tools the agent can call.

Exposed over stdio (MCP) and registered with Codex per-invocation by the agent
(no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.

Most tools are read-only host/task introspection; `tatr_new` is the one write
tool (it creates a tatr task via the `tatr` CLI, bounded to tatr's own tasks
dir). The server runs as a separate trusted process spawned by codex, so this
write is not gated by the model's read-only file sandbox - the curation (fixed
flags, no arbitrary paths) is the guardrail.
"""

from __future__ import annotations

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


_TATR_SORTS = ("created", "priority", "title")


@mcp.tool()
def tatr_ls(filter: str | None = None, sort: str | None = None) -> str:
    """List tatr tasks (one per line: [PRIORITY, TAGS] Title).

    `sort` orders results: "created" (default), "priority" (descending), or
    "title".

    `filter` is a small query over task fields, passed to `tatr ls -f`:
      - fields:      :status  :priority  :tags
      - operators:   eq  contains  in [a, b, ...]
      - connectives: and  or  not, grouped with parentheses
    Examples:
      (:status eq OPEN)
      :tags contains feature
      (:status eq OPEN) and (:tags contains agent)
      :priority eq 0
    """
    args = ["tatr", "ls"]
    if sort:
        if sort not in _TATR_SORTS:
            return f"error: sort must be one of {', '.join(_TATR_SORTS)}"
        args += ["-s", sort]
    if filter:
        args += ["-f", filter]
    return _run(args)


@mcp.tool()
def tatr_show(task_id: str) -> str:
    """Show one tatr task by id (format YYYYMMDD-HHMMSS): status, priority, body."""
    return _run(["tatr", "show", task_id])


@mcp.tool()
def tatr_new(title: str, priority: int = 0, tags: str | None = None) -> str:
    """Create a new tatr task and return its id.

    `priority` is a non-negative integer (higher = more important); `tags` is a
    comma-separated list (e.g. "feature,agent"). The task is created OPEN. IDs are
    second-resolution, so two creates in the same second collide - on an "already
    exists" error, wait a moment and retry.
    """
    title = title.strip()
    if not title:
        return "error: title is required"
    if priority < 0:
        return "error: priority must be a non-negative integer"
    args = ["tatr", "new", title, "-p", str(priority)]
    if tags:
        cleaned = ",".join(t.strip() for t in tags.split(",") if t.strip())
        if cleaned:
            args += ["-t", cleaned]
    return _run(args)


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
    removed = apply_disabled_tools(_disabled_tools())
    if removed:
        logger.info("disabled tools: %s", ", ".join(removed))
    mcp.run()


if __name__ == "__main__":
    main()
