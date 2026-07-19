"""Scufris MCP server: a curated, read-only set of tools the agent can call.

Exposed over stdio (MCP) and registered with Codex per-invocation by the agent
(no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.
"""

from __future__ import annotations

import shutil
import subprocess

from mcp.server.fastmcp import FastMCP

from .metrics import PsutilCollector

# Cap tool output so a huge result can't blow up the model context.
_MAX_OUTPUT = 20_000
_TIMEOUT_SECONDS = 15.0

mcp = FastMCP("scufris")
_collector = PsutilCollector()


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


@mcp.tool()
def tatr_ls(filter: str | None = None) -> str:
    """List tatr tasks. Optional query filter, e.g. '(:status eq OPEN)'."""
    args = ["tatr", "ls"]
    if filter:
        args += ["-f", filter]
    return _run(args)


@mcp.tool()
def tatr_show(task_id: str) -> str:
    """Show one tatr task by id (format YYYYMMDD-HHMMSS)."""
    return _run(["tatr", "show", task_id])


def main() -> None:
    """Run the MCP server over stdio (spawned by Codex)."""
    mcp.run()


if __name__ == "__main__":
    main()
