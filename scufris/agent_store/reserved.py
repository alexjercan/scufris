"""The two synthetic reserved agents, built from settings rather than a row.

The orchestrator and the host agent have no ``agents`` row: their backend and
model come from ``Settings`` and their session id from the session tables, so a
record for either is BUILT on demand instead of loaded. Their live run-state is
the one thing that is not persisted anywhere, so the store holds it in memory -
which is what :class:`ReservedAgents` is: the half of ``AgentStore`` that knows
about the synthetic two.

It is a MIXIN because ``AgentStore`` has one public surface and is at the
600-line source cap; it is not an extension point, and nothing else inherits it.
"""

from __future__ import annotations

from ..config import Settings, canonical_backend, default_model_for
from ..enums import (
    HOST_AGENT_ID,
    ORCHESTRATOR_ID,
    AgentState,
    PermissionMode,
)
from .records import AgentRecord
from .registry import SessionRows

_ORCHESTRATOR_DESCRIPTION = (
    "The landing orchestrator - a default agent that runs in the server "
    "directory and keeps multiple sessions."
)
_HOST_AGENT_DESCRIPTION = (
    "The host agent - bound to this MACHINE rather than to a project. It holds "
    "the host toolset (inspection plus the propose-only actions) and works "
    "through propose -> preview -> approve; it cannot approve anything itself."
)


def orch_backend(settings: Settings) -> str:
    """The orchestrator's effective backend (it tracks the landing settings, so
    its registry entry is keyed by whatever the settings say NOW)."""
    return canonical_backend(settings.agent_backend)


def orchestrator_record(
    settings: Settings, session_id: str | None, state: AgentState
) -> AgentRecord:
    """Build the synthetic reserved orchestrator from settings (never persisted).
    Its backend/model track the landing config."""
    backend = orch_backend(settings)
    return AgentRecord(
        id=ORCHESTRATOR_ID,
        name="Orchestrator",
        project_id="",  # no project binding -> runs in the server cwd
        backend=backend,
        model=default_model_for(settings, backend),
        description=_ORCHESTRATOR_DESCRIPTION,
        session_id=session_id,
        state=state,
        permission_mode=settings.agent_permission_mode,
    )


def host_record(
    settings: Settings, session_id: str | None, state: AgentState
) -> AgentRecord:
    """Build the synthetic reserved HOST agent from settings (never persisted).

    Bound to the machine: no project, so it runs in the server cwd like the
    orchestrator. Its backend/model track the landing config for the same reason
    the orchestrator's do - a second backend setting the operator has to remember
    is a worse deployment than one that follows.

    Its permission mode is fixed at MANUAL (read-only) rather than following
    ``agent_permission_mode``: this agent's job is to read the host and PROPOSE
    changes to it, and every change it can make goes through the helper after an
    operator approval. File-write or unattended-command power would only widen
    what a prompt-injected turn could do, and would buy it nothing it needs.
    """
    backend = orch_backend(settings)
    return AgentRecord(
        id=HOST_AGENT_ID,
        name="Host",
        project_id="",  # bound to the machine -> runs in the server cwd
        backend=backend,
        model=default_model_for(settings, backend),
        description=_HOST_AGENT_DESCRIPTION,
        session_id=session_id,
        state=state,
        permission_mode=PermissionMode.MANUAL,
    )


class ReservedAgents:
    """``AgentStore``'s reserved-agent half: their live state and their records.

    The two reserved agents' run-state is the ONE thing this package does not
    persist: their config comes from settings and their session ids and outcomes
    have their own tables, so a row for them would hold nothing but a lifecycle
    value that a restart is entitled to forget.
    """

    _settings: Settings

    def __init__(self) -> None:
        self._orch_state: AgentState = AgentState.IDLE
        self._host_state: AgentState = AgentState.IDLE

    def _orch_backend(self) -> str:
        return orch_backend(self._settings)

    def _orchestrator_record(self, sessions: SessionRows) -> AgentRecord:
        backend = self._orch_backend()
        return orchestrator_record(
            self._settings,
            sessions.get(ORCHESTRATOR_ID, backend),
            self._orch_state,
        )

    def _host_record(self, sessions: SessionRows) -> AgentRecord:
        backend = self._orch_backend()
        return host_record(
            self._settings,
            sessions.get(HOST_AGENT_ID, backend),
            self._host_state,
        )

    def _reserved_record(
        self, sessions: SessionRows, agent_id: str
    ) -> AgentRecord | None:
        """The synthetic record for a reserved id, or None for a normal agent."""
        if agent_id == ORCHESTRATOR_ID:
            return self._orchestrator_record(sessions)
        if agent_id == HOST_AGENT_ID:
            return self._host_record(sessions)
        return None

    def _set_reserved_state(self, agent_id: str, state: AgentState) -> None:
        if agent_id == ORCHESTRATOR_ID:
            self._orch_state = state
        else:
            self._host_state = state
