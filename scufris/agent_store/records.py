"""The agent record, its id rules, and the errors the store raises."""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..enums import (
    HOST_AGENT_ID,
    ORCHESTRATOR_ID,
    AgentState,
    PermissionMode,
)

# ``AgentStore`` defines a public ``list()`` method, which shadows the builtin
# ``list`` inside class-scope annotations (mypy resolves ``list[str]`` there to the
# method). This module-level alias, bound where ``list`` is still the builtin, lets
# those methods annotate a real list return.
SessionIdList = list[str]

# An agent id is a path/URL segment (`/api/agents/<id>`), so restrict it to a
# safe charset - no slashes, dots or whitespace (mirrors PROJECT_ID_RE).
AGENT_ID_RE = r"^[A-Za-z0-9_-]+$"

# Lifecycle states an agent moves through; the run machinery drives them.
# `AgentLifecycle` is kept as an alias of the shared `AgentState` enum so existing
# importers/annotations keep working.
AgentLifecycle = AgentState


class AgentNotFound(KeyError):
    """Raised when an agent id does not exist."""


class InvalidAgent(ValueError):
    """Raised for an invalid field (empty name, unknown project, bad backend)."""


class AgentsReadOnly(RuntimeError):
    """Raised when a write is attempted while ``settings_writable`` is false."""


class ReservedAgent(RuntimeError):
    """Raised for a mutation not allowed on a RESERVED agent - the orchestrator or
    the host agent (delete, or a config edit that belongs to the settings store).
    Both are synthetic: there is no agents.json row to edit or remove."""


# The reserved, undeletable agents: synthetic records (not in agents.json) whose
# backend/model come from settings, running in the server cwd (no project).
# `ORCHESTRATOR_ID` / `HOST_AGENT_ID` are defined in `enums` (the audience taxonomy
# is derived from them) and re-exported here, where every existing importer looks
# for them.
RESERVED_AGENT_IDS = frozenset({ORCHESTRATOR_ID, HOST_AGENT_ID})


class AgentRecord(BaseModel):
    """A configured agent. ``session_id``/``state`` are set by the run machinery,
    not the CRUD API. ``session_id`` is registry-owned (``SessionRegistry``,
    sessions.json): never persisted with the record, attached at read time."""

    id: str
    name: str
    project_id: str
    backend: str
    model: str = ""
    description: str = ""
    # Retired from the create flow (work is driven by chatting); kept as optional
    # metadata for back-compat with older records.
    goal: str = ""
    task_id: str = ""
    session_id: str | None = None
    state: AgentState = AgentState.IDLE
    permission_mode: PermissionMode = PermissionMode.MANUAL


def _slugify(name: str) -> str:
    """A URL-safe slug from a name (mirrors ``projects._slugify``): lowercase,
    non-alnum -> '-', trimmed; non-ASCII dropped; empty -> ``"agent"``. The output
    is provably confined to ``[A-Za-z0-9_-]`` so it can never carry a
    slash/dot/traversal."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug or "agent"
