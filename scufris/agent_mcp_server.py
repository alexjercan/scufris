"""Scufris `agent` MCP server: the sub-agent callback tools.

A SEPARATE MCP server (id ``agent``) from the orchestrator agentic server
(``mcp_server``, id ``scufris``) and the den life server (``den_mcp_server``,
id ``den``). This is the ONLY scufris server a regular (non-orchestrator)
sub-agent turn registers, so the sub-agent surface is capability-free by
construction: a sub-agent can signal back but can never create/run/observe
agents or inspect the host, because those servers are simply never registered
on its turn (not filtered out at runtime).

The two tools let a sub-agent hand control back to the orchestrator:
``request_input`` (blocked, needs a decision) and ``report_back`` (finished,
here is the result). Both POST to the dashboard's own HTTP API, addressed by
the sub-agent's own id (``SCUFRIS_AGENT_ID``, injected when the dashboard spawns
this server). Steering (``sessions.AGENT_STEERING_PREAMBLE``) refers to these by
bare name, so the distinct ``agent`` server id does not affect it.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .mcp_common import _api_call

logger = logging.getLogger(__name__)

mcp = FastMCP("agent")


def _self_agent_id() -> str:
    """This sub-agent's own id (``SCUFRIS_AGENT_ID``, injected by the dashboard when
    it spawns this server), so the callbacks can address the caller back to the
    API."""
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


@mcp.tool()
def report_back(summary: str) -> str:
    """Signal that you have FINISHED your assigned task and hand back a short result
    summary (what you did, and how it turned out - e.g. "implemented X; all tests
    green"). Records your report and returns immediately - END YOUR TURN right after
    calling this; the orchestrator is woken (or sees you in pending_agents), reads
    your report and acknowledges it. Use this when your work is done, instead of
    ending silently - it is the completion sibling of ``request_input`` (which is for
    when you are BLOCKED and need a decision)."""
    s = summary.strip()
    if not s:
        return "error: summary is required"
    agent_id = _self_agent_id()
    if not agent_id:
        return "error: report_back is unavailable (no agent id in environment)"
    return _api_call("POST", f"/api/agents/{agent_id}/report_back", body={"summary": s})


def main() -> None:
    """Run the sub-agent callback MCP server over stdio (spawned by a backend on a
    sub-agent turn). Configures its own logging - a separate process from the
    dashboard. No disabled-tools filter: the callbacks are not operator-hidable."""
    import os

    from .logsetup import configure_logging

    configure_logging(os.environ.get("SCUFRIS_LOG_LEVEL", "INFO"))
    mcp.run()


if __name__ == "__main__":
    main()
