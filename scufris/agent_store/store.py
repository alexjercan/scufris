"""``AgentStore``: the persisted list of agents and their run lifecycle.

An agent record is the orchestrator's unit of work: a named agent, bound to a
project, with a backend, model, an optional goal or tatr task, and a lifecycle
``state``. This module owns only the STORE; actually RUNNING an agent (launching
a supervised turn, setting ``session_id``/``state``) belongs to the supervisor.

Persistence mirrors ``projects.py`` / ``settings_store.py``: one JSON file under
the state dir, atomic write, tolerant load, writes gated by ``settings_writable``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from ..config import (
    Settings,
    available_backends,
    canonical_backend,
    default_model_for,
    normalize_permission_mode,
)
from ..enums import HOST_AGENT_ID, ORCHESTRATOR_ID, AgentState
from ..projects import ProjectNotFound, ProjectStore
from .outcomes import OutcomeStore, RunOutcome
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
from .registry import SessionRegistry
from .reserved import host_record, orch_backend, orchestrator_record

logger = logging.getLogger(__name__)


class AgentStore:
    """Owns the persisted list of agents (and their session registry)."""

    def __init__(self, settings: Settings, projects: ProjectStore) -> None:
        self._settings = settings
        self._projects = projects
        self._path = Path(settings.state_dir) / "agents.json"
        self._agents: dict[str, AgentRecord] = {}
        # Session ids live in the registry (sessions.json), NOT on the records:
        # persisted for every agent, orchestrator included, so a restart cannot
        # lose the orchestrator's conversation.
        self._registry = SessionRegistry(settings)
        # The durable run-outcome record (final message + terminal state), for
        # every agent - the substrate the orchestrator polls after a run ends,
        # since the per-run EventBus is gone by then.
        self._outcomes = OutcomeStore(settings)
        # The reserved agents' live run-state stays in memory (their config comes
        # from settings; they have no agents.json row).
        self._orch_state: AgentState = AgentState.IDLE
        self._host_state: AgentState = AgentState.IDLE
        self._load()

    def _orch_backend(self) -> str:
        return orch_backend(self._settings)

    def _orchestrator_record(self) -> AgentRecord:
        backend = self._orch_backend()
        return orchestrator_record(
            self._settings,
            self._registry.get(ORCHESTRATOR_ID, backend),
            self._orch_state,
        )

    def _host_record(self) -> AgentRecord:
        backend = self._orch_backend()
        return host_record(
            self._settings,
            self._registry.get(HOST_AGENT_ID, backend),
            self._host_state,
        )

    def _reserved_record(self, agent_id: str) -> AgentRecord | None:
        """The synthetic record for a reserved id, or None for a normal agent."""
        if agent_id == ORCHESTRATOR_ID:
            return self._orchestrator_record()
        if agent_id == HOST_AGENT_ID:
            return self._host_record()
        return None

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _require_writable(self) -> None:
        if not self.writable:
            raise AgentsReadOnly("agents are read-only on this server")

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("agent store: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, list):
            return
        for item in data:
            # Migrate a legacy `write_enabled` bool to a permission mode before
            # validating (the field is gone from the model, so it would be
            # ignored otherwise and a write-enabled agent would silently become
            # read-only).
            if isinstance(item, dict) and "permission_mode" not in item:
                if "write_enabled" in item:
                    item["permission_mode"] = (
                        "edit" if item.get("write_enabled") else "manual"
                    )
            try:
                agent = AgentRecord.model_validate(item)
            except ValueError as exc:
                logger.warning("agent store: dropping invalid record: %s", exc)
                continue
            # Normalize legacy backend ids (codex modes) to the canonical name;
            # persists on the next write.
            agent.backend = canonical_backend(agent.backend)
            # Migrate a pre-registry `session_id` persisted on the record into
            # the registry (once - an existing mapping wins), then drop it from
            # the in-memory record: the registry is the only home of session
            # ids, and `get`/`list` re-attach them at read time.
            if agent.session_id and not self._registry.has(agent.id):
                self._registry.set(agent.id, agent.backend, agent.session_id)
            agent.session_id = None
            self._agents[agent.id] = agent

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # `session_id` is registry-owned (sessions.json); never write it here.
        payload = [a.model_dump(exclude={"session_id"}) for a in self._agents.values()]
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def _with_session(self, agent: AgentRecord) -> AgentRecord:
        """The record as the API sees it: its current session id attached from
        the registry, keyed by the agent's CURRENT backend (a cross-backend id
        reads as None)."""
        return agent.model_copy(
            update={"session_id": self._registry.get(agent.id, agent.backend)}
        )

    def list(self) -> list[AgentRecord]:
        # The reserved ORCHESTRATOR is a HIDDEN default: it is NOT in the list
        # (reached via `/` and `get(ORCHESTRATOR_ID)`, not the /agents grid).
        #
        # The reserved HOST agent IS listed, because the opposite of hidden is
        # what it needs to be: the orchestrator delegates a host change to it by
        # id, so `list_agents` has to show it exists, and the operator should see
        # the agent that can propose changes to their machine sitting in the grid
        # rather than having to know a URL. It is not in `self._agents`, so it is
        # prepended to the persisted ones (and refuses every CRUD mutation).
        return [self._host_record()] + sorted(
            (self._with_session(a) for a in self._agents.values()),
            key=lambda a: a.name.lower(),
        )

    def get(self, agent_id: str) -> AgentRecord:
        reserved = self._reserved_record(agent_id)
        if reserved is not None:
            return reserved
        try:
            return self._with_session(self._agents[agent_id])
        except KeyError as exc:
            raise AgentNotFound(agent_id) from exc

    def _unique_id(self, base: str) -> str:
        if base not in self._agents:
            return base
        n = 2
        while f"{base}-{n}" in self._agents:
            n += 1
        return f"{base}-{n}"

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
        agent = AgentRecord(
            id=self._unique_id(base),
            name=name,
            project_id=project_id,
            backend=backend,
            # An explicit non-empty model wins; anything else (omitted or blank)
            # falls back to the backend's default so a claude agent never keeps
            # a codex model.
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
        self._agents[agent.id] = agent
        self._persist()
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
            # agents.json row).
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
        agent = self._raw(agent_id)
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
        # The model follows the EFFECTIVE backend so a switch never keeps a stale
        # model (e.g. claude showing "gpt-5.5"). An explicit non-empty model
        # wins; a blank one, or a backend change with no model sent, re-defaults.
        eff_backend = backend if backend is not None else agent.backend
        if model is not None:
            updates["model"] = model.strip() or default_model_for(
                self._settings, eff_backend
            )
        elif backend_changed:
            updates["model"] = default_model_for(self._settings, eff_backend)
        # Sessions are BACKEND-SPECIFIC (a codex rollout id means nothing to
        # claude, and vice versa), so a backend switch starts a fresh
        # conversation: clear the registry mapping and reset the run state.
        # The registry's backend key already makes the stale id unreadable, but
        # clearing keeps sessions.json free of dead entries (and switching BACK
        # must not resurrect the old conversation).
        if backend_changed:
            self._registry.clear(agent_id)
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
        self._agents[agent_id] = updated
        self._persist()
        return self._with_session(updated)

    def delete(self, agent_id: str) -> None:
        self._require_writable()
        if agent_id in RESERVED_AGENT_IDS:
            raise ReservedAgent(f"the {agent_id} agent cannot be deleted")
        if agent_id not in self._agents:
            raise AgentNotFound(agent_id)
        del self._agents[agent_id]
        # Drop the session mapping AND the run outcome with the agent, so a
        # future agent that happens to reuse the freed id can never inherit this
        # conversation or a stale "needs input" outcome.
        self._registry.clear(agent_id)
        self._outcomes.clear(agent_id)
        self._persist()

    # --- run-state mutators (driven by the run engine, NOT the CRUD API) ------
    # These persist the lifecycle the Supervisor drives; they are not gated by
    # `_require_writable` because they record server-internal run progress, not a
    # user config edit.

    def mark_running(self, agent_id: str) -> AgentRecord:
        """Record that a run for this agent has started."""
        if agent_id == ORCHESTRATOR_ID:
            self._orch_state = AgentState.RUNNING
            return self._orchestrator_record()
        if agent_id == HOST_AGENT_ID:
            self._host_state = AgentState.RUNNING
            return self._host_record()
        agent = self._raw(agent_id)
        updated = agent.model_copy(update={"state": AgentState.RUNNING})
        self._agents[agent_id] = updated
        self._persist()
        return self._with_session(updated)

    def _raw(self, agent_id: str) -> AgentRecord:
        """The internal record (session_id always None - the registry owns it);
        raises AgentNotFound. Mutators build on this so a registry-attached id
        never leaks back into `self._agents`."""
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise AgentNotFound(agent_id) from exc

    def set_orchestrator_session(self, session_id: str | None) -> None:
        """Switch to (session_id) or start a fresh conversation (None) for the
        orchestrator, in the persisted registry (keyed by the current settings
        backend). "New chat" (None) clears only the CURRENT pointer and KEEPS the
        session history so the switcher still lists prior chats; the next turn's
        minted id is appended by ``mark_finished``. A backend change
        starts a fresh history (sessions are backend-specific)."""
        self._registry.set_current(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_sessions(self) -> SessionIdList:
        """Every session the registry attributes to the orchestrator under its
        current backend, for the switcher list. Ownership is recorded,
        never inferred from a provider disk scan - so a sub-agent's session can
        never appear here."""
        return self._registry.sessions_for(ORCHESTRATOR_ID, self._orch_backend())

    def forget_orchestrator_session(self, session_id: str) -> None:
        """Drop one session from the orchestrator's switcher history (a session
        delete), clearing the current pointer if it was that id."""
        self._registry.remove(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_session_id(self) -> str | None:
        """The orchestrator's current active session id (or None for fresh).
        Registry-backed: it survives a restart, and a stale id recorded under a
        different backend reads as None instead of being resumed."""
        return self._registry.get(ORCHESTRATOR_ID, self._orch_backend())

    def record_running_session(
        self, agent_id: str, backend: str, session_id: str
    ) -> None:
        """Record a run's session id in the registry AS SOON AS it is known
        (turn-start), so a mid-turn refresh sees the session before the terminal
        ``mark_finished``. Works for the orchestrator and sub-agents alike, keyed
        under the LAUNCH-TIME ``backend`` snapshot (the same reasoning as
        ``mark_finished``'s ``backend`` param: a mid-run backend switch must not
        mislabel it). Idempotent - ``mark_finished`` re-setting the same id later is
        a no-op re-current, no double history entry (``registry.set`` dedups)."""
        self._registry.set(agent_id, backend, session_id)

    def record_spawn_parent(
        self,
        child_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Record which agent + orchestrator session spawned ``child_id``,
        so a child's ``request_input`` can be routed back to the chat that launched
        it. Backend-independent; safe before the child has ever run."""
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
        """Record a run's terminal state and (if produced) its session id. The
        session id goes to the registry - for EVERY agent, orchestrator
        included - keyed by the backend the run ACTUALLY executed under. The
        final ``message`` + terminal ``state`` are also written to the durable
        outcome store, so the orchestrator can observe a finished agent after
        the per-run EventBus has closed.

        Pass ``backend`` (the launch-time snapshot's backend) so a backend
        switch that lands mid-run cannot mislabel the finishing session:
        without it we would re-read the now-current backend and record the old
        session id under the wrong label, defeating the registry's
        backend-mismatch guard. Omitted -> the agent's current backend, correct
        whenever no switch raced the turn."""
        # Coerce a raw string to the enum: `model_copy(update=...)` below does NOT
        # validate, so a str here would settle on the AgentState field unconverted
        # and later trip pydantic's enum serializer.
        state = AgentState(state)
        # If `request_input` (WAITING) or `report_back` (REPORTED) fired during THIS
        # run, the agent ended its turn deliberately with a signal - the natural DONE
        # that follows must not clobber it. Preserve the same-run, unacknowledged
        # signal outcome (and its message), refreshing only the now-finalized session
        # id. Keyed on run_id so a signal left by an EARLIER run is still overwritten
        # by a later run's completion, and an ERROR still wins (a crash is neither a
        # clean wait nor a clean report). The outcome is built now but WRITTEN only
        # once the agent is known to EXIST (after `_raw` / the orchestrator branch),
        # so a delete-mid-run cannot resurrect it.
        # Only a natural DONE preserves a same-run WAITING/REPORTED signal. A
        # non-DONE terminal (ERROR, or a user CANCELLED) intentionally SUPERSEDES an
        # unacknowledged same-run signal: if the user stops a run that had emitted
        # request_input, the explicit stop wins over the now-moot pending question.
        existing = self._outcomes.get(agent_id)
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
            outcome = existing.model_copy(
                update={
                    "session_id": session_id or existing.session_id,
                    "ts": time.time(),
                }
            )
            eff_state = existing.state
        else:
            outcome = RunOutcome(
                state=state,
                message=message,
                run_id=run_id,
                session_id=session_id,
                ts=time.time(),
                acknowledged=False,
            )
            eff_state = state
        if agent_id in RESERVED_AGENT_IDS:
            # A reserved agent's run-state is in-memory (it has no agents.json
            # row); only its session id persists, via the registry.
            if agent_id == ORCHESTRATOR_ID:
                self._orch_state = eff_state
            else:
                self._host_state = eff_state
            if session_id is not None:
                self._registry.set(
                    agent_id, backend or self._orch_backend(), session_id
                )
            self._outcomes.set(agent_id, outcome)
            record = self._reserved_record(agent_id)
            assert record is not None  # narrowed by RESERVED_AGENT_IDS
            return record
        agent = self._raw(agent_id)
        if session_id is not None:
            self._registry.set(agent_id, backend or agent.backend, session_id)
        self._outcomes.set(agent_id, outcome)
        updated = agent.model_copy(update={"state": eff_state})
        self._agents[agent_id] = updated
        self._persist()
        return self._with_session(updated)

    def _require_exists(self, agent_id: str) -> None:
        """Existence guard for the signal mutators, which write an outcome rather
        than an agents.json row.

        The HOST agent has no row and must still be able to signal: outcomes are
        keyed by agent id in their own store, so nothing here depends on the row
        existing. `_raw` is still the guard for a normal agent, so a delete racing a
        live sub-agent's callback writes nothing (the
        completion-callback-write-after-existence-check lesson).

        The ORCHESTRATOR is deliberately NOT exempt, even though it is equally
        synthetic: it registers no `agent` callback server, so it has no way to
        signal and no reason to - and a route that accepted its id would be
        accepting a caller that cannot exist.
        """
        if agent_id == HOST_AGENT_ID:
            return
        self._raw(agent_id)

    def awaiting_approval(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that an agent has proposed a host action and is waiting for the
        OPERATOR to decide (see ``OutcomeStore.awaiting_approval``)."""
        self._require_exists(agent_id)
        return self._outcomes.awaiting_approval(
            agent_id, summary, run_id=run_id, session_id=session_id
        )

    def request_input(
        self,
        agent_id: str,
        question: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that a (mid-run) agent needs a decision (see
        ``OutcomeStore.request_input``). Raises AgentNotFound for a missing agent
        (the caller is a live sub-agent, but a delete could race), writing nothing
        in that case."""
        self._require_exists(agent_id)  # raises before any write
        return self._outcomes.request_input(
            agent_id, question, run_id=run_id, session_id=session_id
        )

    def report_back(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that a (mid-run) agent has finished and reported a result (see
        ``OutcomeStore.report_back``). Raises AgentNotFound for a missing agent
        (the caller is a live sub-agent, but a delete could race), writing nothing
        in that case."""
        self._require_exists(agent_id)  # raises before any write
        return self._outcomes.report_back(
            agent_id, summary, run_id=run_id, session_id=session_id
        )

    def outcome(self, agent_id: str) -> RunOutcome | None:
        """The agent's most-recent durable run outcome, or None if it has not
        finished a run yet."""
        return self._outcomes.get(agent_id)

    def outcomes(self) -> dict[str, RunOutcome]:
        """All agents' most-recent run outcomes, keyed by agent id. The
        orchestrator's "who needs me" poll reads from here."""
        return self._outcomes.all()

    def pending_outcomes(self) -> dict[str, RunOutcome]:
        """The agents with an UNACKNOWLEDGED signal (see
        ``OutcomeStore.pending``)."""
        return self._outcomes.pending()

    def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's outcome handled so it drops out of `pending_outcomes`
        (see ``OutcomeStore.acknowledge``)."""
        return self._outcomes.acknowledge(agent_id)
