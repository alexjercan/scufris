"""Which scufris MCP servers a turn registers, and codex's rendering of them.

``scufris_mcp_servers`` is the ONE source of that decision; every backend formats
its result to its own flavour, so codex and claude can never drift on which
servers and env a turn exposes. The audience split is PHYSICAL - a sub-agent
simply has no orchestrator server on its turn - rather than a per-server role
filter, so a backend only ever allow-lists a registered server whole.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from ..auth import API_TOKEN_ENV
from ..config import Settings
from ..enums import Audience, audience_for


def _server_override(
    server_id: str,
    command: str,
    args: list[str],
    approve: bool,
    env: dict[str, str] | None = None,
) -> list[str]:
    """The `-c` lines registering one MCP server for a codex invocation."""
    out = ["-c", f"mcp_servers.{server_id}.command={json.dumps(command)}"]
    if args:
        out += ["-c", f"mcp_servers.{server_id}.args={json.dumps(args)}"]
    if approve:
        out += [
            "-c",
            f'mcp_servers.{server_id}.default_tools_approval_mode="approve"',
        ]
    for key, value in (env or {}).items():
        out += ["-c", f"mcp_servers.{server_id}.env.{key}={json.dumps(value)}"]
    return out


@dataclass(frozen=True)
class ScufrisMcpServer:
    """One backend-agnostic scufris MCP server registration for a turn: its id, the
    process to launch, and the env that configures it.

    A turn can register SEVERAL of these (an orchestrator turn gets ``scufris`` +
    ``den``; a sub-agent turn gets only ``agent``). Each backend formats them to
    its own flavour - codex to ``-c mcp_servers.<id>.*`` overrides
    (``_mcp_overrides``), claude to a ``--mcp-config`` JSON blob
    (``backends.claude._scufris_claude_args``) - from this ONE source, so the two
    can never drift on which servers/env a turn exposes. The audience split is
    PHYSICAL (which servers are on the turn), not a per-server role filter, so a
    backend only allow-lists each registered server whole.
    """

    server_id: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def scufris_mcp_servers(
    settings: Settings,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
    orch_session_id: str = "",
) -> list[ScufrisMcpServer]:
    """The scufris MCP servers to register for this turn (possibly empty).

    The audience split is PHYSICAL, not a runtime filter (``enums.audience_for``
    decides which audience a turn is):

    - an ORCHESTRATOR turn registers the ``scufris`` agentic server plus the
      ``den`` life server (``den`` only when a den is configured);
    - the HOST AGENT's turn registers the ``host`` server - the host toolset
      including the mutating propose tools, which no other audience has - plus the
      ``agent`` callback server, so it can report back and be resumed like any
      other sub-agent;
    - a regular sub-AGENT turn (``agent_id`` set) registers ONLY the ``agent``
      callback server, so it can never reach the orchestrator/den/host tools
      because those servers are simply not on its turn.

    Returns ``[]`` when tools are disabled, or for a turn with no identity at all
    (nothing to address the callbacks back to).

    ``orch_session_id`` is the orchestrator's CURRENT session (the id this turn is
    resuming), injected as ``SCUFRIS_ORCH_SESSION_ID`` on the ``scufris`` server so
    ``message_agent`` / ``run_agent`` can stamp a spawned child with the chat that
    launched it and ``pending_agents`` can route escalations back to it. Empty on a
    fresh turn (no resumed id yet) - the child is then unattributed.
    """
    if not settings.agent_tools_enabled:
        return []
    api_base = f"http://{settings.host}:{settings.port}"
    command = sys.executable
    disabled = ",".join(settings.disabled_tools) if settings.disabled_tools else ""
    # The machine credential for the dashboard's own HTTP API, minted per process
    # by create_app onto ITS settings object (never os.environ - see
    # `Settings.auth_api_token`). Only the servers that CALL the API carry it
    # (`scufris` and the sub-agent `agent` callback server) - the den server does
    # not talk to the API, so it has no business holding a credential for it.
    # Empty when no app is running (a bare `scufris mcp-server` for probing),
    # which simply means the tools authenticate with nothing and are refused by a
    # gated dashboard.
    api_token = settings.auth_api_token
    servers: list[ScufrisMcpServer] = []
    audience = audience_for(is_orchestrator=is_orchestrator, agent_id=agent_id)

    def callback_server() -> ScufrisMcpServer:
        """The ``agent`` callback server, addressed to this agent."""
        agent_env = {"SCUFRIS_API_BASE": api_base, "SCUFRIS_AGENT_ID": agent_id}
        if api_token:
            agent_env[API_TOKEN_ENV] = api_token
        return ScufrisMcpServer(
            "agent", command, ("-m", "scufris.agent_mcp_server"), agent_env
        )

    if audience is Audience.ORCHESTRATOR:
        scufris_env: dict[str, str] = {"SCUFRIS_API_BASE": api_base}
        if api_token:
            scufris_env[API_TOKEN_ENV] = api_token
        if orch_session_id:
            scufris_env["SCUFRIS_ORCH_SESSION_ID"] = orch_session_id
        if disabled:
            scufris_env["SCUFRIS_DISABLED_TOOLS"] = disabled
        servers.append(
            ScufrisMcpServer(
                "scufris", command, ("-m", "scufris.mcp_server"), scufris_env
            )
        )
        # The den (`the-den`) server is orchestrator-only AND opt-in: registered
        # only when a den is configured, and ONLY it carries the den path, so a
        # project sub-agent can never reach the operator's journal. The operator's
        # disabled-tool set applies here too (den tools are hidable).
        if settings.den_path is not None:
            den_env = {"SCUFRIS_DEN_PATH": str(settings.den_path)}
            if disabled:
                den_env["SCUFRIS_DISABLED_TOOLS"] = disabled
            servers.append(
                ScufrisMcpServer(
                    "den", command, ("-m", "scufris.den_mcp_server"), den_env
                )
            )
    elif audience is Audience.HOST:
        # The host toolset, including the propose tools no other audience has. It
        # calls the dashboard's API (to propose and to read the queue), so it
        # carries the machine credential and its own agent id - which is how a
        # proposal is audited as coming from this agent rather than from "an
        # agent". It cannot approve with that credential: the decision endpoints
        # refuse it outright (`auth.OPERATOR_ONLY_PATTERN`).
        host_env: dict[str, str] = {
            "SCUFRIS_API_BASE": api_base,
            "SCUFRIS_AGENT_ID": agent_id,
        }
        if api_token:
            host_env[API_TOKEN_ENV] = api_token
        if disabled:
            host_env["SCUFRIS_DISABLED_TOOLS"] = disabled
        servers.append(
            ScufrisMcpServer(
                "host", command, ("-m", "scufris.host_mcp_server"), host_env
            )
        )
        # Plus the ordinary callbacks: the host agent reports back and is resumed
        # through the SAME machinery as any sub-agent, rather than a second
        # communication path of its own.
        servers.append(callback_server())
    elif audience is Audience.AGENT:
        servers.append(callback_server())
    return servers


def _mcp_overrides(
    settings: Settings,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
    orch_session_id: str = "",
) -> list[str]:
    """`-c` config registering the MCP servers for this invocation.

    Injected on the `codex app-server` argv so nothing is written to `~/.codex`.
    The built-in scufris servers come from the shared ``scufris_mcp_servers`` core
    (an orchestrator turn gets ``scufris`` + ``den``; a sub-agent turn gets only
    ``agent``), so codex and claude never drift on which servers a turn exposes;
    codex formats each to `-c mcp_servers.<id>.*` overrides here. The audience
    split is PHYSICAL - a sub-agent simply has no ``scufris``/``den`` server - so a
    regular agent gets no other scufris tools and draws the rest from its project
    config/skills. For an unattended codex run, MCP tool calls
    would otherwise be auto-cancelled (no stdin to approve on), so trusted servers
    auto-approve their tools and approval_policy is never. The sandbox (set per
    turn on thread/start|resume) remains the real guardrail.
    """
    if not settings.agent_tools_enabled:
        return []
    args: list[str] = []
    servers = scufris_mcp_servers(
        settings,
        is_orchestrator=is_orchestrator,
        agent_id=agent_id,
        orch_session_id=orch_session_id,
    )
    for server in servers:
        args += _server_override(
            server.server_id,
            server.command,
            list(server.args),
            approve=True,
            env=server.env,
        )
    args += ["-c", 'approval_policy="never"']
    return args
