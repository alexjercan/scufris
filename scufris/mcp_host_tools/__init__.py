"""The host toolset: the read-only inspection tools and the propose-only actions.

Defined here ONCE and registered onto a server by ``register``, because the two
audiences that may see host tools need different halves of the same set:

- the ORCHESTRATOR (``mcp_server``, server id ``scufris``) registers the
  INSPECTION half, so "why is this box hot" stays a direct answer in the chat the
  operator is already in;
- the HOST AGENT (``host_mcp_server``, server id ``host``) registers inspection
  AND the mutating propose tools, so the propose -> preview -> approve contract is
  stated to exactly one audience and lives in one steering preamble.

The audience split is PHYSICAL: a tool reaches an audience by being registered on
a server that audience's turn wires up, never by a runtime filter. That is why
this package exports functions and a registrar instead of decorating at
definition - the same function object is registered on one server or two,
and nothing has to be filtered out afterwards. It is also why the two halves are
two modules: `inspection` and `actions`.

Every tool that shells out uses a fixed argument list (never a shell string), a
timeout, and bounded output. Nothing here can CHANGE the host: the propose tools
ask the dashboard's API for a preview and leave the action waiting for an
operator, which is the only route to root on this box.

This module is the package's public surface; the submodules import each other
directly rather than through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .actions import (
    host_action_audit,
    host_action_status,
    nixos_change_status,
    propose_host_action,
    propose_nixos_change,
)
from .inspection import (
    _format_processes,
    disk_usage,
    host_failed_units,
    host_flake_status,
    host_generation_diff,
    host_journal,
    host_largest_directories,
    host_network,
    host_reclaimable_space,
    host_stats,
    host_storage,
    host_thermal,
    host_unit_status,
    host_units,
    host_what_provides,
    list_processes,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# The two halves, named so a registration site reads as the audience it serves.
# `INSPECTION` never changes the host; `ACTIONS` can only ever ASK for a change.
INSPECTION: tuple[Callable[..., Any], ...] = (
    host_stats,
    disk_usage,
    list_processes,
    host_units,
    host_failed_units,
    host_unit_status,
    host_journal,
    host_storage,
    host_largest_directories,
    host_reclaimable_space,
    host_network,
    host_thermal,
    host_what_provides,
    host_generation_diff,
    host_flake_status,
)

# Propose-only, and there is deliberately no approve tool in this set (nor
# anywhere else): approving is an operator act gated on a real session by the
# middleware (`auth.OPERATOR_ONLY_PATTERN`), and the HTTP endpoint refuses the
# machine bearer token these subprocesses hold. `tests/test_host_mcp_server.py`
# asserts the absence, so a future convenience tool cannot quietly appear.
ACTIONS: tuple[Callable[..., Any], ...] = (
    propose_host_action,
    host_action_status,
    propose_nixos_change,
    nixos_change_status,
    host_action_audit,
)


def register(mcp: "FastMCP", *, actions: bool) -> None:
    """Register the host tools on ``mcp``: inspection always, the propose tools
    only when ``actions`` is set.

    Each function is registered under its own name and docstring, exactly as an
    ``@mcp.tool()`` decorator would - the difference is only that the audience
    decides, at registration time, which half exists on that server at all.
    """
    for tool in INSPECTION:
        mcp.tool()(tool)
    if actions:
        for tool in ACTIONS:
            mcp.tool()(tool)


__all__ = [
    "ACTIONS",
    "INSPECTION",
    "_format_processes",
    "disk_usage",
    "host_action_audit",
    "host_action_status",
    "host_failed_units",
    "host_flake_status",
    "host_generation_diff",
    "host_journal",
    "host_largest_directories",
    "host_network",
    "host_reclaimable_space",
    "host_stats",
    "host_storage",
    "host_thermal",
    "host_unit_status",
    "host_units",
    "host_what_provides",
    "list_processes",
    "nixos_change_status",
    "propose_host_action",
    "propose_nixos_change",
    "register",
]
