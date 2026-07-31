"""Shared option enums for the stringly-typed fields across the app.

All are ``enum.StrEnum`` so each member IS its string value: pydantic validates
membership on a field typed with one, but JSON/log output and ``==`` comparisons
against the raw string are unchanged (no wire-format change). Centralised here so
config, the agent store, the supervisor and the API models share ONE definition
of each fixed option set instead of duplicating ``Literal[...]`` / bare ``str``.
"""

from __future__ import annotations

from enum import StrEnum

# The two RESERVED, synthetic agent ids: agents that exist without an agents.json
# row, built from settings and bound to no project. They live here, in the
# dependency-free module, because `Audience` below is derived from them and every
# audience dispatch site would otherwise have to import the agent store.
# `agent_store` re-exports both so its existing importers are unaffected.
ORCHESTRATOR_ID = "orchestrator"
# The host agent: bound to the MACHINE rather than to a project, and the only
# audience that carries the mutating host tools.
HOST_AGENT_ID = "host"


class Audience(StrEnum):
    """Which scufris MCP servers - and which steering preamble - a turn gets.

    The split is PHYSICAL: an audience's servers are the ones its turn wires up,
    so a tool it should not have is not registered rather than filtered out at
    call time. Derived from the running agent's id by ``audience_for``, which is
    the ONE place the mapping lives:

    - ``ORCHESTRATOR`` -> ``scufris`` (agentic) + ``den`` (life, when configured)
    - ``HOST`` -> ``host`` (the host toolset, mutating half included) + ``agent``
      (the callbacks, so it can report back and be resumed)
    - ``AGENT`` -> ``agent`` only: a project sub-agent can signal, and nothing more
    - ``NONE`` -> no scufris servers at all (a turn with no identity to address,
      or a tools-disabled turn)
    """

    ORCHESTRATOR = "orchestrator"
    HOST = "host"
    AGENT = "agent"
    NONE = "none"


def audience_for(*, is_orchestrator: bool = False, agent_id: str = "") -> Audience:
    """The audience of a turn, from the identity it runs under.

    ``is_orchestrator`` wins over ``agent_id`` (the landing orchestrator is never a
    regular agent, and a stray id must not demote it). The host agent is
    recognised by its reserved id, so every dispatch site - the server list, the
    codex/claude wiring, the steering preamble, the in-process health probe -
    agrees on what a turn is by construction rather than by repeating the
    comparison.
    """
    if is_orchestrator:
        return Audience.ORCHESTRATOR
    if agent_id == HOST_AGENT_ID:
        return Audience.HOST
    if agent_id:
        return Audience.AGENT
    return Audience.NONE


class AuthMode(StrEnum):
    """How an agent authenticates to its backend CLI. The subscription login
    differs per backend (ChatGPT for codex, claude.ai for claude); ``API_KEY`` is
    the metered-key alternative for either. ``LOCAL`` is the no-login mode for a
    self-hosted endpoint (opencode driving a llama.cpp server)."""

    CHATGPT = "chatgpt"  # codex: Sign in with ChatGPT subscription (primary)
    CLAUDE_AI = "claude_ai"  # claude: Sign in with claude.ai subscription
    API_KEY = "api_key"  # metered API key (either backend)
    LOCAL = "local"  # opencode: self-hosted llama.cpp endpoint, no login


class AuthPolicy(StrEnum):
    """Whether the DASHBOARD requires an authenticated operator session.

    Distinct from ``AuthMode``, which is about how an AGENT authenticates to its
    backend CLI - this one is about who may reach the HTTP surface at all.

    ``AUTO`` resolves from the bind address (open on loopback, mandatory off it),
    so the deployed LAN bind is protected without the operator opting in and
    development stays login-free. ``DISABLED`` is refused on a non-loopback bind
    (``auth.validate_auth_config``); it is a development convenience, not an
    escape hatch.
    """

    AUTO = "auto"
    REQUIRED = "required"
    DISABLED = "disabled"


class Backend(StrEnum):
    """The agent backend an agent (and the landing orchestrator) runs on. Legacy
    codex MODE ids (``app_server``/``exec``) fold to ``codex`` via
    ``config.canonical_backend`` before reaching a field typed with this."""

    CODEX = "codex"
    CLAUDE = "claude"
    OPENCODE = "opencode"  # opencode serve daemon (self-hosted llama.cpp)
    MOCK = "mock"  # in-process dev/test fake (behind enable_mock_backend)


class PermissionMode(StrEnum):
    """An agent's write posture, mapped to a per-backend flag in ``backends`` (codex
    ``--sandbox`` / claude ``--permission-mode``)."""

    MANUAL = "manual"  # read-only: observe/plan
    EDIT = "edit"  # may edit project files
    AUTO = "auto"  # edit + run commands unattended (full access)


class AgentState(StrEnum):
    """An agent's merged lifecycle state (run engine + rollout), for the UI."""

    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"  # waiting on an approval
    WAITING = "waiting"  # ended a turn awaiting a decision (needs orchestrator input)
    REPORTED = "reported"  # ended a turn having reported its result (orchestrator reads + acks)
    DONE = "done"
    ERROR = "error"
    CANCELLED = (
        "cancelled"  # a run the user stopped on purpose (stop button / cancel_agent)
    )


class RunPhase(StrEnum):
    """The supervisor run engine's low-level phase for a single supervised run
    (merged into the richer ``AgentState`` for display)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
