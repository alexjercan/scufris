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

import shutil
import subprocess

from mcp.server.fastmcp import FastMCP

from .metrics import PsutilCollector
from .processes import ProcessList, PsutilProcessCollector

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
        return f"error: {args[0]} not found on PATH"
    try:
        proc = subprocess.run(
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"error: {args[0]} timed out after {timeout}s"
    output = proc.stdout
    if proc.returncode != 0:
        output = (output + "\n" + proc.stderr).strip() or f"exit {proc.returncode}"
    return output[:_MAX_OUTPUT]


@mcp.tool()
def host_stats() -> dict[str, object]:
    """Snapshot of host metrics: CPU, memory, swap, disks, load, network, uptime."""
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
    """Disk usage per real filesystem (df -h), excluding tmpfs/overlay noise."""
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
    """Top running applications by CPU, grouped by name (like a compact htop)."""
    return _format_processes(_proc_collector.sample(), limit)


def main() -> None:
    """Run the MCP server over stdio (spawned by Codex)."""
    mcp.run()


if __name__ == "__main__":
    main()
