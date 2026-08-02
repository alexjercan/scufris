"""``AgentStore``: the persisted list of agents and their run lifecycle.

An agent record is the orchestrator's unit of work: a named agent, bound to a
project, with a backend, model, an optional goal or tatr task, and a lifecycle
``state``. This module owns only the STORE; actually RUNNING an agent (launching
a supervised turn, setting ``session_id``/``state``) belongs to the supervisor.

The store reads THROUGH the database, like ``projects.py``: every method opens
one ``Database.transaction()``, and there is no in-memory mirror to go stale or
to publish a write that failed. Session ids and outcomes live in their own
tables, reached on the SAME connection - which is what makes the completion path
(``mark_finished``) one commit rather than three, so an outcome the orchestrator
can poll never exists without the session record it names.

The class is split across three files because it is at the 600-line source cap:
``reserved`` holds the synthetic-agent half, ``signals`` the mid-run signal half,
and ``rows`` the ``agents``-table reads both need. They are one class, not an
extension point.
"""

from __future__ import annotations

import logging
import re
import time

from sqlalchemy import delete as sql_delete
from sqlalchemy import insert, select
from sqlalchemy import update as sql_update
from sqlalchemy.exc import IntegrityError

from ..config import (
    Settings,
    available_backends,
    canonical_backend,
    default_model_for,
    normalize_permission_mode,
)
from ..db import Database
from ..db.models import AgentRow
from ..enums import HOST_AGENT_ID, ORCHESTRATOR_ID, AgentState
from ..projects import ProjectNotFound, ProjectStore
from .outcomes import OutcomeRows, RunOutcome
from .records import (
    AGENT_ID_RE,
    RESERVED_AGENT_IDS,
    AgentNotFound,
    AgentRecord,
    AgentsReadOnly,
    InvalidAgent,
    ReservedAgent,
    SessionIdList,
    _slugify,
)
from .registry import SessionRegistry, SessionRows
from .reserved import ReservedAgents
from .rows import fetch, record, require, unique_id, with_session
from .signals import AgentSignals

logger = logging.getLogger(__name__)


class AgentStore(AgentSignals, ReservedAgents):
    """Owns the persisted list of agents (and their sessions and outcomes)."""

    def __init__(
        self, settings: Settings, projects: ProjectStore, db: Database
    ) -> None:
        ReservedAgents.__init__(self)
        self._settings = settings
        self._projects = projects
        self._db = db
        # Session ids live in their own tables, NOT on the agent record:
        # persisted for every agent, orchestrator included, so a restart cannot
        # lose the orchestrator's conversation.
        self._registry = SessionRegistry(db)

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _require_writable(self) -> None:
        if not self.writable:
            raise AgentsReadOnly("agents are read-only on this server")

    def list(self) -> list[AgentRecord]:
        # The reserved ORCHESTRATOR is a HIDDEN default: it is NOT in the list
        # (reached via `/` and `get(ORCHESTRATOR_ID)`, not the /agents grid).
        #
        # The reserved HOST agent IS listed, because the opposite of hidden is
        # what it needs to be: the orchestrator delegates a host change to it by
        # id, so `list_agents` has to show it exists, and the operator should see
        # the agent that can propose changes to their machine sitting in the grid
        # rather than having to know a URL. It has no row, so it is prepended to
        # the persisted ones (and refuses every CRUD mutation).
        with self._db.transaction() as conn:
            sessions = SessionRows(conn)
            host = self._host_record(sessions)
            agents = [
                with_session(sessions, record(row))
                for row in conn.execute(select(AgentRow.__table__)).all()
            ]
        # Ordered here rather than in SQL: SQLite's own lower() is ASCII-only,
        # and the ordering the API has always published is Python's.
        return [host] + sorted(agents, key=lambda a: a.name.lower())

    def get(self, agent_id: str) -> AgentRecord:
        with self._db.transaction() as conn:
            sessions = SessionRows(conn)
            reserved = self._reserved_record(sessions, agent_id)
            if reserved is not None:
                return reserved
            agent = fetch(conn, agent_id)
            if agent is None:
                raise AgentNotFound(agent_id)
            return with_session(sessions, agent)

    def create(
        self,
        name: str,
        project_id: str,
        *,
        backend: str | None = None,
        model: str | None = None,
        description: str = "",
        goal: str = "",
        task_id: str = "",
        permission_mode: str = "manual",
    ) -> AgentRecord:
        self._require_writable()
        name = name.strip()
        if not name:
            raise InvalidAgent("agent name must not be empty")
        # Resolved BEFORE the transaction: the project store opens one of its
        # own, and units of work do not nest.
        try:
            self._projects.get(project_id)
        except ProjectNotFound as exc:
            raise InvalidAgent(f"no such project: {project_id}") from exc
        backend = canonical_backend(backend or self._settings.agent_backend)
        allowed = available_backends(self._settings)
        if backend not in allowed:
            raise InvalidAgent(
                f"unknown or disabled backend {backend!r}; available: {allowed}"
            )
        base = _slugify(name)
        if not re.fullmatch(AGENT_ID_RE, base):
            raise InvalidAgent(f"cannot derive a valid id from name {name!r}")
        if base in RESERVED_AGENT_IDS:
            raise InvalidAgent(f"{base!r} is a reserved agent id")
        with self._db.transaction() as conn:
            agent = AgentRecord(
                id=unique_id(conn, base),
                name=name,
                project_id=project_id,
                backend=backend,
                # An explicit non-empty model wins; anything else (omitted or
                # blank) falls back to the backend's default so a claude agent
                # never keeps a codex model.
                model=(
                    (model.strip() or default_model_for(self._settings, backend))
                    if model is not None
                    else default_model_for(self._settings, backend)
                ),
                description=description.strip(),
                goal=goal.strip(),
                task_id=task_id.strip(),
                permission_mode=normalize_permission_mode(permission_mode),
            )
            try:
                conn.execute(insert(AgentRow).values(**_columns(agent)))
            except IntegrityError as exc:
                # The id is a real PRIMARY KEY now. `unique_id` above shares this
                # transaction so nothing outside can win the race, but the
                # constraint is the authority and its violation is a domain
                # error, not a database error surfacing at a route.
                raise InvalidAgent(f"agent id already exists: {agent.id}") from exc
        return agent

    def update(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        backend: str | None = None,
        model: str | None = None,
        description: str | None = None,
        goal: str | None = None,
        task_id: str | None = None,
        permission_mode: str | None = None,
    ) -> AgentRecord:
        self._require_writable()
        if agent_id == ORCHESTRATOR_ID:
            # The orchestrator's config lives in the settings store (it has no
            # `agents` row).
            raise ReservedAgent(
                "the orchestrator is configured from settings, not /api/agents"
            )
        if agent_id == HOST_AGENT_ID:
            # Same reason, plus one of its own: the host agent's read-only
            # permission mode is a safety property of the audience that may
            # propose host changes, not a preference to edit through the API.
            raise ReservedAgent(
                "the host agent is configured from settings, not /api/agents"
            )
        with self._db.transaction() as conn:
            agent = require(conn, agent_id)
            updates: dict[str, object] = {}
            if name is not None:
                name = name.strip()
                if not name:
                    raise InvalidAgent("agent name must not be empty")
                updates["name"] = name
            backend_changed = False
            if backend is not None:
                backend = canonical_backend(backend)
                allowed = available_backends(self._settings)
                if backend not in allowed:
                    raise InvalidAgent(
                        f"unknown or disabled backend {backend!r}; available: {allowed}"
                    )
                updates["backend"] = backend
                backend_changed = backend != agent.backend
            # The model follows the EFFECTIVE backend so a switch never keeps a
            # stale model (e.g. claude showing "gpt-5.5"). An explicit non-empty
            # model wins; a blank one, or a backend change with no model sent,
            # re-defaults.
            eff_backend = backend if backend is not None else agent.backend
            if model is not None:
                updates["model"] = model.strip() or default_model_for(
                    self._settings, eff_backend
                )
            elif backend_changed:
                updates["model"] = default_model_for(self._settings, eff_backend)
            # Sessions are BACKEND-SPECIFIC (a codex rollout id means nothing to
            # claude, and vice versa), so a backend switch starts a fresh
            # conversation: clear the session record and reset the run state. The
            # backend key already makes the stale id unreadable, but clearing
            # keeps the tables free of dead entries (and switching BACK must not
            # resurrect the old conversation).
            if backend_changed:
                SessionRows(conn).clear(agent_id)
                updates["state"] = AgentState.IDLE
            if description is not None:
                updates["description"] = description.strip()
            if goal is not None:
                updates["goal"] = goal.strip()
            if task_id is not None:
                updates["task_id"] = task_id.strip()
            if permission_mode is not None:
                updates["permission_mode"] = normalize_permission_mode(permission_mode)
            updated = agent.model_copy(update=updates)
            conn.execute(
                sql_update(AgentRow)
                .where(AgentRow.id == agent_id)
                .values(**_columns(updated))
            )
            return with_session(SessionRows(conn), updated)

    def delete(self, agent_id: str) -> None:
        self._require_writable()
        if agent_id in RESERVED_AGENT_IDS:
            raise ReservedAgent(f"the {agent_id} agent cannot be deleted")
        with self._db.transaction() as conn:
            result = conn.execute(sql_delete(AgentRow).where(AgentRow.id == agent_id))
            if result.rowcount == 0:
                raise AgentNotFound(agent_id)
            # Drop the session record AND the run outcome with the agent, in the
            # same commit, so a future agent that happens to reuse the freed id
            # can never inherit this conversation or a stale "needs input"
            # outcome.
            SessionRows(conn).clear(agent_id)
            OutcomeRows(conn).clear(agent_id)

    # --- run-state mutators (driven by the run engine, NOT the CRUD API) ------
    # These persist the lifecycle the Supervisor drives; they are not gated by
    # `_require_writable` because they record server-internal run progress, not a
    # user config edit.

    def mark_running(self, agent_id: str) -> AgentRecord:
        """Record that a run for this agent has started."""
        with self._db.transaction() as conn:
            sessions = SessionRows(conn)
            if agent_id in RESERVED_AGENT_IDS:
                self._set_reserved_state(agent_id, AgentState.RUNNING)
                reserved = self._reserved_record(sessions, agent_id)
                assert reserved is not None  # narrowed by RESERVED_AGENT_IDS
                return reserved
            agent = require(conn, agent_id)
            updated = agent.model_copy(update={"state": AgentState.RUNNING})
            conn.execute(
                sql_update(AgentRow)
                .where(AgentRow.id == agent_id)
                .values(state=AgentState.RUNNING.value)
            )
            return with_session(sessions, updated)

    def set_orchestrator_session(self, session_id: str | None) -> None:
        """Switch to (session_id) or start a fresh conversation (None) for the
        orchestrator, in the persisted session tables (keyed by the current
        settings backend). "New chat" (None) clears only the CURRENT pointer and
        KEEPS the session history so the switcher still lists prior chats; the
        next turn's minted id is appended by ``mark_finished``. A backend change
        starts a fresh history (sessions are backend-specific)."""
        self._registry.set_current(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_sessions(self) -> SessionIdList:
        """Every session attributed to the orchestrator under its current
        backend, for the switcher list. Ownership is recorded, never inferred
        from a provider disk scan - so a sub-agent's session can never appear
        here."""
        return self._registry.sessions_for(ORCHESTRATOR_ID, self._orch_backend())

    def forget_orchestrator_session(self, session_id: str) -> None:
        """Drop one session from the orchestrator's switcher history (a session
        delete), clearing the current pointer if it was that id."""
        self._registry.remove(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_session_id(self) -> str | None:
        """The orchestrator's current active session id (or None for fresh).
        Row-backed: it survives a restart, and a stale id recorded under a
        different backend reads as None instead of being resumed."""
        return self._registry.get(ORCHESTRATOR_ID, self._orch_backend())

    def record_running_session(
        self, agent_id: str, backend: str, session_id: str
    ) -> None:
        """Record a run's session id AS SOON AS it is known (turn-start), so a
        mid-turn refresh sees the session before the terminal ``mark_finished``.
        Works for the orchestrator and sub-agents alike, keyed under the
        LAUNCH-TIME ``backend`` snapshot (the same reasoning as
        ``mark_finished``'s ``backend`` param: a mid-run backend switch must not
        mislabel it). Idempotent - ``mark_finished`` re-setting the same id later
        is a no-op re-current, no double history entry."""
        self._registry.set(agent_id, backend, session_id)

    def record_spawn_parent(
        self,
        child_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Record which agent + orchestrator session spawned ``child_id``,
        so a child's ``request_input`` can be routed back to the chat that
        launched it. Backend-independent; safe before the child has ever run."""
        self._registry.set_parent(child_id, parent_agent_id, parent_session_id)

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        """The ``(parent_agent_id, parent_session_id)`` that spawned ``agent_id``,
        or ``(None, None)`` for an unattributed agent (UI-spawned, or spawned in a
        fresh orchestrator turn before its session id existed)."""
        return self._registry.parent_of(agent_id)

    def mark_finished(
        self,
        agent_id: str,
        *,
        state: AgentState,
        session_id: str | None = None,
        backend: str | None = None,
        message: str = "",
        run_id: str = "",
    ) -> AgentRecord:
        """Record a run's terminal state and (if produced) its session id, in ONE
        transaction.

        The three writes - the agent row's terminal state, the session record,
        and the outcome - commit together or not at all. That is the guarantee
        this store exists for: the orchestrator polls outcomes, so an outcome
        without the session record it names is a report of a conversation that
        cannot be resumed.

        Pass ``backend`` (the launch-time snapshot's backend) so a backend switch
        that lands mid-run cannot mislabel the finishing session: without it we
        would re-read the now-current backend and record the old session id under
        the wrong label, defeating the backend-mismatch guard. Omitted -> the
        agent's current backend, correct whenever no switch raced the turn."""
        # Coerce a raw string to the enum: `model_copy(update=...)` below does
        # NOT validate, so a str here would settle on the AgentState field
        # unconverted and later trip pydantic's enum serializer.
        state = AgentState(state)
        with self._db.transaction() as conn:
            sessions = SessionRows(conn)
            outcomes = OutcomeRows(conn)
            outcome, eff_state = self._terminal_outcome(
                outcomes, agent_id, state, session_id, message, run_id
            )
            if agent_id in RESERVED_AGENT_IDS:
                # A reserved agent's run-state is in-memory (it has no `agents`
                # row); only its session id and outcome persist.
                self._set_reserved_state(agent_id, eff_state)
                if session_id is not None:
                    sessions.add(agent_id, backend or self._orch_backend(), session_id)
                outcomes.set(agent_id, outcome)
                reserved = self._reserved_record(sessions, agent_id)
                assert reserved is not None  # narrowed by RESERVED_AGENT_IDS
                return reserved
            # The outcome is WRITTEN only once the agent is known to EXIST, so a
            # delete-mid-run cannot resurrect it.
            agent = require(conn, agent_id)
            if session_id is not None:
                sessions.add(agent_id, backend or agent.backend, session_id)
            outcomes.set(agent_id, outcome)
            updated = agent.model_copy(update={"state": eff_state})
            conn.execute(
                sql_update(AgentRow)
                .where(AgentRow.id == agent_id)
                .values(state=eff_state.value)
            )
            return with_session(sessions, updated)

    def _terminal_outcome(
        self,
        outcomes: OutcomeRows,
        agent_id: str,
        state: AgentState,
        session_id: str | None,
        message: str,
        run_id: str,
    ) -> tuple[RunOutcome, AgentState]:
        """The outcome a finishing run should leave, and the state that goes with it.

        If `request_input` (WAITING), `report_back` (REPORTED) or a host proposal
        (BLOCKED) fired during THIS run, the agent ended its turn deliberately
        with a signal - the natural DONE that follows must not clobber it.
        Preserve the same-run, unacknowledged signal outcome (and its message),
        refreshing only the now-finalized session id. Keyed on run_id so a signal
        left by an EARLIER run is still overwritten by a later run's completion.

        Only a natural DONE preserves a signal. A non-DONE terminal (ERROR, or a
        user CANCELLED) intentionally SUPERSEDES an unacknowledged same-run
        signal: if the user stops a run that had emitted request_input, the
        explicit stop wins over the now-moot pending question.

        The existing outcome is read on the CALLER'S connection, so the read and
        the write that follows it are one unit of work - the read-modify-write
        window this used to have is closed.
        """
        existing = outcomes.get(agent_id)
        preserve_signal = (
            state == AgentState.DONE
            and existing is not None
            and bool(existing.run_id)
            and existing.run_id == run_id
            and existing.state
            in (AgentState.WAITING, AgentState.REPORTED, AgentState.BLOCKED)
            and not existing.acknowledged
        )
        if preserve_signal:
            assert existing is not None  # narrowed by preserve_signal
            return (
                existing.model_copy(
                    update={
                        "session_id": session_id or existing.session_id,
                        "ts": time.time(),
                    }
                ),
                existing.state,
            )
        return (
            RunOutcome(
                state=state,
                message=message,
                run_id=run_id,
                session_id=session_id,
                ts=time.time(),
                acknowledged=False,
            ),
            state,
        )


def _columns(agent: AgentRecord) -> dict[str, object]:
    """The record as ``agents`` columns: everything but the registry-owned
    ``session_id``, with the two enums as their stored string values."""
    return agent.model_dump(mode="json", exclude={"session_id"})
