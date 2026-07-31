"""First-class agents: a configured agent bound to a project, persisted to a
state file.

An agent record is the orchestrator's unit of work: a named agent, bound to a
project, with a backend, model, an optional goal or tatr task and a lifecycle
``state``. This package owns the store and its records; actually RUNNING an agent
is the supervisor's job.

Four pieces sit behind this facade: ``records`` (the record, its id rules and the
errors), ``registry`` (the only home of session ids and session ownership),
``outcomes`` (a run's terminal result, kept past the per-run EventBus) and
``store`` (``AgentStore`` itself, plus ``reserved`` for the two synthetic agents).

Named ``agent_store`` (not ``agents``) to avoid confusion with ``agent``'s
``Agent`` protocol.
"""

from __future__ import annotations

from ..enums import HOST_AGENT_ID, ORCHESTRATOR_ID, AgentState
from .outcomes import OutcomeStore, RunOutcome
from .records import (
    AGENT_ID_RE,
    RESERVED_AGENT_IDS,
    AgentLifecycle,
    AgentNotFound,
    AgentRecord,
    AgentsReadOnly,
    InvalidAgent,
    ReservedAgent,
    SessionIdList,
)
from .registry import SessionRegistry
from .store import AgentStore

__all__ = [
    "AGENT_ID_RE",
    "RESERVED_AGENT_IDS",
    "AgentLifecycle",
    "AgentNotFound",
    "AgentRecord",
    "AgentState",
    "AgentStore",
    "AgentsReadOnly",
    "HOST_AGENT_ID",
    "InvalidAgent",
    "ORCHESTRATOR_ID",
    "OutcomeStore",
    "ReservedAgent",
    "RunOutcome",
    "SessionIdList",
    "SessionRegistry",
]
