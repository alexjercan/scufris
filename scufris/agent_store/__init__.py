"""First-class agents: a configured agent bound to a project, in the state database.

An agent record is the orchestrator's unit of work: a named agent, bound to a
project, with a backend, model, an optional goal or tatr task and a lifecycle
``state``. This package owns the store and its records; actually RUNNING an agent
is the supervisor's job.

Behind this facade: ``records`` (the record, its id rules and the errors),
``registry`` (the only home of session ids and session ownership), ``outcomes``
(a run's terminal result, kept past the per-run EventBus), and ``AgentStore``
itself - which is one class split across ``store``, ``reserved`` (the two
synthetic agents) and ``signals`` (the mid-run signal mutators), over the
``agents``-table reads in ``rows``.

Each of the three tables has a ``*Rows`` class that works on an OPEN
``Connection`` as well as a ``Database``-owning wrapper. That is what lets the
completion path write the agent row, the session record and the outcome in ONE
transaction: units of work do not nest on this engine, so a caller that is
already inside one passes its connection down.

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
