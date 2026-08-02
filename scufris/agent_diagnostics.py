"""What any one agent can report about itself, asked backend-first.

The scoped ``/api/agents/{id}/*`` diagnostics - account, usage, memory, health,
visible tools and MCP server health - used to resolve capability by comparing a
canonical backend NAME at each call site, so a fifth adapter would have had to
find every one of them. Here the question is asked of the agent's own backend
adapter instead (``read_usage``, ``read_memory_footprint``, ``has_scufris_mcp``),
and the answer comes back in a ``Capability`` envelope that distinguishes "this
backend has no such reader" from "the reader found nothing".

The service is transport-independent: it takes an already-resolved
``AgentRecord`` and raises nothing HTTP-shaped, so the 404 for an unknown agent
stays in the route and the Telegram bot can call the same readers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .agent_store import AgentRecord
from .backends import Capability, get_backend
from .config import Settings, auth_mode_for_backend
from .enums import ORCHESTRATOR_ID, AuthMode
from .health import AgentHealth, agent_health
from .mcp_health import probe_server, servers_for_audience
from .mcp_models import AgentTool, McpServerHealth, ToolParam
from .sessions import MemoryFootprint, UsageQuota


class AccountInfo(BaseModel):
    """The account backing the agent, for the console's Account panel."""

    # None for a backend with no login (mock); else the backend's auth mode
    # (codex -> chatgpt/api_key, claude -> claude_ai/api_key).
    auth_mode: AuthMode | None
    model: str
    enabled: bool
    quota: Capability[UsageQuota]


def tool_parameters(input_schema: object) -> list[ToolParam]:
    """Distill a tool's JSON ``inputSchema`` into typed params for the runner.

    Reads ``properties`` (name -> {type, description, default}) and the top-level
    ``required`` list. Unknown/missing types fall back to "string" so the form
    still renders a text input. Best-effort: a malformed schema yields [].
    """
    if not isinstance(input_schema, dict):
        return []
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return []
    required = input_schema.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    params: list[ToolParam] = []
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        raw_type = spec.get("type")
        params.append(
            ToolParam(
                name=str(name),
                type=raw_type if isinstance(raw_type, str) else "string",
                required=name in required_set,
                description=str(spec.get("description") or ""),
                default=spec.get("default"),
            )
        )
    return params


def _as_agent_tool(t: Any, server: str, disabled: set[str]) -> AgentTool:
    schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
    props = schema.get("properties")
    args = list(props) if isinstance(props, dict) else []
    return AgentTool(
        name=t.name,
        description=t.description or "",
        server=server,
        args=args,
        parameters=tool_parameters(t.inputSchema),
        enabled=t.name not in disabled,
    )


def mcp_servers_for_audience(agent_id: str) -> list[tuple[str, Any]]:
    """The in-process ``(server_id, FastMCP)`` pairs for an agent's audience:
    the orchestrator's ``scufris`` + ``den``, the host agent's ``host`` +
    ``agent``, or a regular sub-agent's ``agent`` server (from
    ``mcp_health.servers_for_audience``, which mirrors what a real turn
    registers)."""
    return servers_for_audience(agent_id == ORCHESTRATOR_ID, agent_id)


async def tools_for_servers(
    settings: Settings, servers: list[tuple[str, Any]]
) -> list[AgentTool]:
    """Aggregate the tools of the given in-process ``(server_id, FastMCP)`` pairs,
    each tool tagged with its real server id and its enabled flag from the
    operator disabled-tool set. Mirrors what a real turn registers, so the
    read-only listing matches what the audience actually gets."""
    disabled = set(settings.disabled_tools)
    out: list[AgentTool] = []
    for server_id, mcp in servers:
        for t in await mcp.list_tools():
            out.append(_as_agent_tool(t, server_id, disabled))
    return out


async def probe_servers(
    settings: Settings, servers: list[tuple[str, Any]]
) -> list[McpServerHealth]:
    """Live-probe each server (``mcp_health.probe_server``) into an
    ``McpServerHealth`` for the settings "MCP tools" section: the server's
    status/detail plus its tools, each tool's ``available`` flag set from the
    server's probe verdict and ``enabled`` from the operator disabled-tool set.
    The den path must already be bridged by the caller so the den readiness check
    sees it."""
    disabled = set(settings.disabled_tools)
    out: list[McpServerHealth] = []
    for server_id, mcp in servers:
        status, detail, available, tools = await probe_server(server_id, mcp, disabled)
        agent_tools: list[AgentTool] = []
        for t in tools:
            at = _as_agent_tool(t, server_id, disabled)
            at.available = available
            agent_tools.append(at)
        out.append(
            McpServerHealth(
                id=server_id, status=status, detail=detail, tools=agent_tools
            )
        )
    return out


class AgentDiagnostics:
    """Backend-aware read-only diagnostics for one agent at a time.

    Every method takes the agent's RESOLVED record, so the backend, model and auth
    mode - and therefore the whole capability set - follow the persisted record. A
    backend switch on the orchestrator moves all of them together, because they
    are all asked of ``get_backend(agent.backend)``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def usage(self, agent: AgentRecord) -> Capability[UsageQuota]:
        """The account-level usage/quota behind this agent's backend."""
        return get_backend(agent.backend).read_usage(self._settings)

    def memory(self, agent: AgentRecord) -> Capability[MemoryFootprint]:
        """This agent's backend's persistent on-disk footprint."""
        return get_backend(agent.backend).read_memory_footprint(self._settings)

    def account(self, agent: AgentRecord) -> AccountInfo:
        """The account backing THIS agent: its effective model, auth mode and (when
        the backend has a reader) its usage quota - all off the agent's record."""
        return AccountInfo(
            auth_mode=auth_mode_for_backend(self._settings, agent.backend),
            model=agent.model,
            enabled=self._settings.agent_enabled,
            quota=self.usage(agent),
        )

    async def health(self, agent: AgentRecord) -> AgentHealth:
        """Read-only diagnostics probed for THIS agent's backend (a claude agent
        probes the claude CLI, not codex), with the MCP rows scoped to its
        audience. Bridge the den path first so the den probe sees it."""
        return await agent_health(
            self._settings,
            backend=agent.backend,
            is_orchestrator=agent.id == ORCHESTRATOR_ID,
            agent_id=agent.id,
            has_scufris_mcp=get_backend(agent.backend).has_scufris_mcp,
        )

    async def tools(self, agent: AgentRecord) -> Capability[list[AgentTool]]:
        """The scufris MCP tools THIS agent can call in its turns - audience- and
        backend-scoped. Unsupported when the agent's backend wires no scufris MCP:
        it has no listing to give, which is not the same as an empty one."""
        if not get_backend(agent.backend).has_scufris_mcp:
            return Capability.unsupported()
        servers = mcp_servers_for_audience(agent.id)
        return Capability.read(await tools_for_servers(self._settings, servers))

    async def mcp(self, agent: AgentRecord) -> list[McpServerHealth]:
        """Live per-server health for THIS agent's audience; empty when the agent's
        backend wires no scufris MCP. Bridge the den path first."""
        if not get_backend(agent.backend).has_scufris_mcp:
            return []
        return await probe_servers(self._settings, mcp_servers_for_audience(agent.id))
