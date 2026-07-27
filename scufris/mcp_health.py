"""In-process live health probe for the scufris MCP servers.

The Health card shows one per-server status row (via ``health.agent_health``) and
the settings "MCP tools" section groups the tools by server; both use this probe.
Rather than spawn each server the way a turn does, the dashboard probes them the
SAME way it already lists and runs their tools: in-process, by importing each
module's ``mcp`` and calling ``list_tools()``, plus each server's real readiness
checks. This catches the failure modes that matter
- an import/list error, or the ``den`` server with no den configured or its
``today``/``macros`` CLIs missing - without the cost and inconsistency of a
subprocess handshake. See ``tasks/20260727-105609/DECISION.md``.

``servers_for_audience`` mirrors ``agent.scufris_mcp_servers`` (which servers a
real turn registers), so the probe covers exactly the audience's servers: an
orchestrator's ``scufris`` + ``den``, or a sub-agent's ``agent`` callback server.
"""

from __future__ import annotations

import os
import shutil
from typing import Any


def servers_for_audience(is_orchestrator: bool) -> list[tuple[str, Any]]:
    """The in-process ``(server_id, FastMCP)`` pairs to probe/list for an audience:
    the orchestrator's ``scufris`` + ``den`` servers, or a sub-agent's ``agent``
    callback server. Imports lazily so a probe never pulls every server module
    unless it is asked for that audience."""
    if is_orchestrator:
        from . import den_mcp_server, mcp_server

        return [("scufris", mcp_server.mcp), ("den", den_mcp_server.mcp)]
    from . import agent_mcp_server

    return [("agent", agent_mcp_server.mcp)]


def _den_readiness(n_tools: int, n_disabled: int) -> tuple[str, str, bool]:
    """The ``den`` server's readiness beyond just advertising tools: the journal
    tools need ``SCUFRIS_DEN_PATH`` configured and present, and both the ``today``
    and ``macros`` CLIs on PATH; any of those missing means the tools are listed
    but cannot actually run (amber, not available). Reads the SAME env the den
    tools read - the dashboard bridges it via ``app._ensure_den_path``."""
    den = os.environ.get("SCUFRIS_DEN_PATH", "").strip()
    if not den:
        return "warn", "the-den not configured (SCUFRIS_DEN_PATH unset)", False
    if not os.path.isdir(os.path.expanduser(den)):
        return "warn", f"den path does not exist: {den}"[:120], False
    missing = [cli for cli in ("today", "macros") if shutil.which(cli) is None]
    if missing:
        return "warn", f"CLI not on PATH: {', '.join(missing)}", False
    if n_disabled:
        return "warn", f"{n_disabled} tool(s) disabled", True
    return "ok", f"{n_tools} tools ready", True


async def probe_server(
    server_id: str, mcp: Any, disabled: set[str]
) -> tuple[str, str, bool, list[Any]]:
    """Live-probe ONE in-process scufris MCP server.

    Returns ``(status, detail, available, tools)``: ``status`` is ok | warn | error
    (green / amber / red dot), ``detail`` a short summary, ``available`` whether the
    server's tools can actually run (False on error or a den that cannot work), and
    the raw tool objects from ``list_tools`` (empty on error). A probe never raises
    - a failure IS the result.
    """
    try:
        tools = list(await mcp.list_tools())
    except Exception as exc:  # noqa: BLE001 - a probe reports failure, never raises
        return "error", f"list_tools failed: {exc}"[:120], False, []
    if not tools:
        return "error", "no tools advertised", False, []
    n_disabled = len({t.name for t in tools} & disabled)
    if server_id == "den":
        status, detail, available = _den_readiness(len(tools), n_disabled)
    elif n_disabled:
        status, detail, available = "warn", f"{n_disabled} tool(s) disabled", True
    else:
        status, detail, available = "ok", f"{len(tools)} tools ready", True
    return status, detail, available, tools
