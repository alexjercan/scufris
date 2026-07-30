"""Scufris `host` MCP server: the host agent's toolset.

A SEPARATE MCP server (id ``host``) from the orchestrator agentic server
(``mcp_server``, id ``scufris``), the den life server (``den_mcp_server``, id
``den``) and the sub-agent callback server (``agent_mcp_server``, id ``agent``).
Only a HOST-AGENT turn registers this one, alongside the ``agent`` callback
server - so the mutating host tools exist for exactly one audience, and the
propose -> preview -> approve contract is stated in exactly one steering
preamble (``tasks/20260729-125040/DECISION.md`` section 2).

The surface: the whole host toolset from ``mcp_host_tools`` - the read-only
inspection half (which the orchestrator also has) PLUS the propose-only mutating
half (``propose_host_action``, ``propose_nixos_change``, their status readers and
``host_action_audit``), which the orchestrator no longer has.

There is deliberately NO approve tool here either. This server holds the power to
ASK, and nothing else: approving is an operator act gated on a real session by
the middleware (``auth.OPERATOR_ONLY_PATTERN``), and the decision endpoints
refuse the machine bearer token this subprocess carries. Being the audience that
may propose does not make an agent the audience that may decide.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .mcp_common import _disabled_tools, apply_disabled_tools
from .mcp_host_tools import register as register_host_tools

logger = logging.getLogger(__name__)

mcp = FastMCP("host")
register_host_tools(mcp, actions=True)


def main() -> None:
    """Run the host agent's MCP server over stdio (spawned by a backend).

    A separate process from the dashboard, so it configures its own logging from
    ``SCUFRIS_LOG_LEVEL`` (to stderr; the backend captures it). The operator
    disabled-tool set applies here as it does to the orchestrator's servers: a
    tool the operator switched off is removed from the registry before this
    server answers anything, so it is never advertised or callable.
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
