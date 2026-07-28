"""MCP tool DTOs shared across the API and the Telegram frontend.

These describe the orchestrator's tool surface for the read-only settings views:
one tool (``AgentTool``), one of its input parameters (``ToolParam``), and one
MCP server's live-probe result (``McpServerHealth``). They live here - not in
``app.py`` - so a frontend that renders them (the Telegram bot) can import them
without a cycle back through ``app`` (which imports ``telegram``). ``app.py``
re-exports the names, so the API/OpenAPI shape is unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel


class ToolParam(BaseModel):
    """One input parameter of an MCP tool, distilled from its JSON input schema.

    The "try it" runner (settings page) generates a form field from each param:
    ``type`` picks the input kind (text/number/checkbox), ``required`` marks it.
    """

    name: str
    type: str = "string"  # JSON-schema type: string/integer/number/boolean/...
    required: bool = False
    description: str = ""
    default: object | None = None


class AgentTool(BaseModel):
    name: str
    description: str
    server: str = "scufris"  # the MCP server that exposes it
    args: list[str] = []  # parameter names, from the tool's input schema
    parameters: list[ToolParam] = []  # full param schema, for the "try it" runner
    enabled: bool = True  # False when the operator disabled it (disabled_tools)
    available: bool = True  # False when its server is unhealthy (probe verdict)


class McpServerHealth(BaseModel):
    """One scufris MCP server's live-probe result for the settings "MCP tools"
    section. ``status`` is ok | warn | error (green / amber / red dot); ``tools``
    are its tools each with an ``enabled`` (operator toggle) and ``available``
    (server reachable) flag driving the per-tool bulb."""

    id: str  # the server id (scufris | den | agent)
    status: str  # "ok" | "warn" | "error"
    detail: str  # short human-readable summary of the probe verdict
    tools: list[AgentTool] = []
