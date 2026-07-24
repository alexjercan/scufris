"""First-class agents: a configured agent bound to a project, persisted to a
state file.

An agent record is the orchestrator's unit of work (tasks/20260720-221748/
SPIKE.md revision 1): a named agent, bound to a project, with a backend, model,
an optional goal or tatr task, a lifecycle ``state`` and a per-agent ``write``
opt-in. This module owns only the STORE + its records; actually RUNNING an agent
(launching a supervised turn, setting ``session_id``/``state``) is A3.

Persistence mirrors ``projects.py`` / ``settings_store.py``: one JSON file under
the state dir, atomic write, tolerant load, writes gated by ``settings_writable``.
Named ``agent_store`` (not ``agents``) to avoid confusion with ``agent.py``'s
``Agent`` protocol.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import (
    Settings,
    available_backends,
    canonical_backend,
    default_model_for,
    normalize_permission_mode,
)
from .enums import AgentState, PermissionMode
from .projects import ProjectNotFound, ProjectStore

logger = logging.getLogger(__name__)

# ``AgentStore`` defines a public ``list()`` method, which shadows the builtin
# ``list`` inside class-scope annotations (mypy resolves ``list[str]`` there to the
# method). This module-level alias, bound where ``list`` is still the builtin, lets
# those methods annotate a real list return.
SessionIdList = list[str]

# An agent id is a path/URL segment (`/api/agents/<id>`), so restrict it to a
# safe charset - no slashes, dots or whitespace (mirrors PROJECT_ID_RE).
AGENT_ID_RE = r"^[A-Za-z0-9_-]+$"

# Lifecycle states an agent moves through; the run machinery (A3) drives them.
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
    """Raised for a mutation not allowed on the reserved orchestrator agent
    (delete, or - in B5a - a config edit that belongs to the settings store)."""


# The reserved, undeletable orchestrator: a synthetic agent (not in agents.json)
# whose backend/model come from settings. It runs in the server cwd (no project).
ORCHESTRATOR_ID = "orchestrator"
_ORCHESTRATOR_DESCRIPTION = (
    "The landing orchestrator - a default agent that runs in the server "
    "directory and (from B5c) keeps multiple sessions."
)


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


class SessionRegistry:
    """The persisted `(agent_id -> backend session history)` mapping - the ONLY
    home of session ids AND session ownership, for ALL agents (the orchestrator
    included).

    Each entry is
    ``{backend, session_id (current | None), sessions: [id,...], parent_agent_id}``.
    It records the backend the ids belong to, because a session id is
    backend-specific (a codex rollout id means nothing to claude - task
    20260721-152034): every accessor returns nothing on a backend mismatch, so a
    stale cross-backend id is structurally unreachable, and a backend switch
    starts a fresh history. ``sessions`` is the full set of sessions the agent
    has owned under ``backend`` - the switcher lists from THIS, so it never has
    to infer ownership from a provider disk scan (part 1, spike
    20260724-111839). ``parent_agent_id`` is reserved for part 3 (who spawned
    this agent); it is stored and preserved but not yet used here.

    Persistence mirrors the other stores (one JSON file under the state dir,
    atomic write, tolerant load - including the legacy ``{backend, session_id}``
    shape, which loads as a one-element history). Not gated by
    ``settings_writable``: like the run-state mutators, it records
    server-internal run progress, not a user config edit."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.state_dir) / "sessions.json"
        self._sessions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("session registry: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        for agent_id, entry in data.items():
            if not (isinstance(agent_id, str) and isinstance(entry, dict)):
                continue
            backend = entry.get("backend")
            if not isinstance(backend, str):
                continue
            session_id = entry.get("session_id")
            session_id = session_id if isinstance(session_id, str) else None
            raw_sessions = entry.get("sessions")
            if isinstance(raw_sessions, list):
                sessions = [s for s in raw_sessions if isinstance(s, str)]
            elif session_id is not None:
                sessions = [session_id]  # legacy {backend, session_id} shape
            else:
                sessions = []
            parent = entry.get("parent_agent_id")
            parent_session = entry.get("parent_session_id")
            self._sessions[agent_id] = {
                "backend": backend,
                "session_id": session_id,
                "sessions": sessions,
                "parent_agent_id": parent if isinstance(parent, str) else None,
                "parent_session_id": (
                    parent_session if isinstance(parent_session, str) else None
                ),
            }

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._sessions, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def _entry(self, agent_id: str, backend: str) -> dict[str, Any] | None:
        """The agent's entry IF it belongs to ``backend``, else None (so a
        cross-backend id is unreachable everywhere, not just in ``get``)."""
        entry = self._sessions.get(agent_id)
        if entry is None or entry["backend"] != backend:
            return None
        return entry

    def _fresh(self, agent_id: str, backend: str, session_id: str | None) -> None:
        """Replace the agent's entry with a fresh history under ``backend``,
        preserving the spawn parent (agent + session) if one was recorded - the
        parent is a fact about who spawned the child, independent of which backend
        it later runs under, so a backend switch must not drop it."""
        prev = self._sessions.get(agent_id)
        self._sessions[agent_id] = {
            "backend": backend,
            "session_id": session_id,
            "sessions": [session_id] if session_id else [],
            "parent_agent_id": prev.get("parent_agent_id") if prev else None,
            "parent_session_id": prev.get("parent_session_id") if prev else None,
        }

    def get(self, agent_id: str, backend: str) -> str | None:
        """The agent's current session id under ``backend``, or None when there
        is no mapping or the stored ids belong to another backend."""
        entry = self._entry(agent_id, backend)
        return entry["session_id"] if entry is not None else None

    def sessions_for(self, agent_id: str, backend: str) -> list[str]:
        """The agent's full session history under ``backend`` (the switcher list),
        or ``[]`` on a backend mismatch / no mapping."""
        entry = self._entry(agent_id, backend)
        return list(entry["sessions"]) if entry is not None else []

    def has(self, agent_id: str) -> bool:
        """Whether ANY mapping exists for this agent (backend-agnostic; the
        legacy-migration guard)."""
        return agent_id in self._sessions

    def set(self, agent_id: str, backend: str, session_id: str) -> None:
        """Back-compat alias of ``add`` (append a minted session + re-current)."""
        self.add(agent_id, backend, session_id)

    def add(self, agent_id: str, backend: str, session_id: str) -> None:
        """Record a newly-minted session: set it current AND append it to the
        history (deduped). A backend change starts a fresh history."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            self._fresh(agent_id, backend, session_id)
        else:
            entry["session_id"] = session_id
            if session_id not in entry["sessions"]:
                entry["sessions"].append(session_id)
        self._persist()

    def set_current(
        self, agent_id: str, backend: str, session_id: str | None
    ) -> None:
        """Switch to (or, with None, clear) the current session WITHOUT dropping
        history - this is "new chat" / "switch chat". A switched-to id not yet in
        the history is appended; a backend change starts a fresh history."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            self._fresh(agent_id, backend, session_id)
        else:
            entry["session_id"] = session_id
            if session_id and session_id not in entry["sessions"]:
                entry["sessions"].append(session_id)
        self._persist()

    def remove(self, agent_id: str, backend: str, session_id: str) -> None:
        """Drop one session from the agent's history (a session delete), clearing
        ``current`` if it was that id. No-op on a backend mismatch / unknown id."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            return
        changed = False
        if session_id in entry["sessions"]:
            entry["sessions"].remove(session_id)
            changed = True
        if entry["session_id"] == session_id:
            entry["session_id"] = None
            changed = True
        if changed:
            self._persist()

    def clear(self, agent_id: str) -> None:
        if self._sessions.pop(agent_id, None) is not None:
            self._persist()

    def set_parent(
        self,
        agent_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Record which agent + session spawned ``agent_id`` (part 3). Parent is a
        backend-independent fact, so this works even when the child has no session
        entry yet: a minimal placeholder is created (backend "") and later upgraded
        in place by ``_fresh`` when the child actually runs (which preserves the
        parent). Persists."""
        entry = self._sessions.get(agent_id)
        if entry is None:
            entry = {
                "backend": "",
                "session_id": None,
                "sessions": [],
                "parent_agent_id": None,
                "parent_session_id": None,
            }
            self._sessions[agent_id] = entry
        entry["parent_agent_id"] = parent_agent_id
        entry["parent_session_id"] = parent_session_id
        self._persist()

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        """The ``(parent_agent_id, parent_session_id)`` recorded for ``agent_id``,
        or ``(None, None)``. Backend-agnostic (parent is not session-specific)."""
        entry = self._sessions.get(agent_id)
        if entry is None:
            return (None, None)
        return (entry.get("parent_agent_id"), entry.get("parent_session_id"))


class RunOutcome(BaseModel):
    """The durable terminal outcome of an agent's most recent run: the final
    message + terminal state, persisted PAST the ephemeral per-run EventBus so
    the orchestrator can observe a finished agent later (BC1, spike
    20260723-001256). ``acknowledged`` lets the orchestrator mark an outcome
    handled (BC3) so it stops showing up as pending."""

    state: AgentState
    message: str = ""
    run_id: str = ""
    session_id: str | None = None
    ts: float = 0.0
    acknowledged: bool = False


class OutcomeStore:
    """The persisted `(agent_id -> most-recent run outcome)` mapping - the
    durable substitute for the per-run EventBus, which closes when a run ends.
    Mirrors ``SessionRegistry``: one JSON file under the state dir, atomic write,
    tolerant load. Not gated by ``settings_writable`` - like the run-state
    mutators it records server-internal run progress, not a user config edit."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.state_dir) / "outcomes.json"
        self._outcomes: dict[str, RunOutcome] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("outcome store: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        for agent_id, entry in data.items():
            if not (isinstance(agent_id, str) and isinstance(entry, dict)):
                continue
            try:
                self._outcomes[agent_id] = RunOutcome.model_validate(entry)
            except ValueError as exc:
                logger.warning("outcome store: dropping invalid outcome: %s", exc)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {aid: o.model_dump() for aid, o in self._outcomes.items()}
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def get(self, agent_id: str) -> RunOutcome | None:
        return self._outcomes.get(agent_id)

    def all(self) -> dict[str, RunOutcome]:
        return dict(self._outcomes)

    def set(self, agent_id: str, outcome: RunOutcome) -> None:
        self._outcomes[agent_id] = outcome
        self._persist()

    def clear(self, agent_id: str) -> None:
        if self._outcomes.pop(agent_id, None) is not None:
            self._persist()


class AgentStore:
    """Owns the persisted list of agents (and their session registry)."""

    def __init__(self, settings: Settings, projects: ProjectStore) -> None:
        self._settings = settings
        self._projects = projects
        self._path = Path(settings.state_dir) / "agents.json"
        self._agents: dict[str, AgentRecord] = {}
        # Session ids live in the registry (sessions.json), NOT on the records:
        # persisted for every agent, orchestrator included, so a restart cannot
        # lose the orchestrator's conversation (bug 20260723-001251).
        self._registry = SessionRegistry(settings)
        # The durable run-outcome record (final message + terminal state), for
        # every agent - the substrate the orchestrator polls after a run ends,
        # since the per-run EventBus is gone by then (BC1, spike 20260723-001256).
        self._outcomes = OutcomeStore(settings)
        # The orchestrator's live run-state stays in memory (its config comes
        # from settings; it has no agents.json row).
        self._orch_state: AgentState = AgentState.IDLE
        self._load()

    def _orch_backend(self) -> str:
        """The orchestrator's effective backend (it tracks the landing settings,
        so its registry entry is keyed by whatever the settings say NOW)."""
        return canonical_backend(self._settings.agent_backend)

    def _orchestrator_record(self) -> AgentRecord:
        """Build the synthetic reserved orchestrator from settings (never
        persisted). Its backend/model track the landing config."""
        backend = self._orch_backend()
        return AgentRecord(
            id=ORCHESTRATOR_ID,
            name="Orchestrator",
            project_id="",  # no project binding -> runs in the server cwd
            backend=backend,
            model=default_model_for(self._settings, backend),
            description=_ORCHESTRATOR_DESCRIPTION,
            session_id=self._registry.get(ORCHESTRATOR_ID, backend),
            state=self._orch_state,
            permission_mode=self._settings.agent_permission_mode,
        )

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
        # The reserved orchestrator is a HIDDEN default: it is NOT in the list
        # (reached via `/` and `get(ORCHESTRATOR_ID)`, not the /agents grid), so
        # `list()` returns only the real, project-bound agents.
        return sorted(
            (self._with_session(a) for a in self._agents.values()),
            key=lambda a: a.name.lower(),
        )

    def get(self, agent_id: str) -> AgentRecord:
        if agent_id == ORCHESTRATOR_ID:
            return self._orchestrator_record()
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
        if base == ORCHESTRATOR_ID:
            raise InvalidAgent(f"{ORCHESTRATOR_ID!r} is a reserved agent id")
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
            # agents.json row); the editable seam lands in B5b. See the task.
            raise ReservedAgent(
                "the orchestrator is configured from settings, not /api/agents"
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
        # must not resurrect the old conversation). See task 20260721-152034.
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
        if agent_id == ORCHESTRATOR_ID:
            raise ReservedAgent("the orchestrator agent cannot be deleted")
        if agent_id not in self._agents:
            raise AgentNotFound(agent_id)
        del self._agents[agent_id]
        # Drop the session mapping AND the run outcome with the agent, so a
        # future agent that happens to reuse the freed id can never inherit this
        # conversation or a stale "needs input" outcome.
        self._registry.clear(agent_id)
        self._outcomes.clear(agent_id)
        self._persist()

    # --- run-state mutators (used by the run engine, A3; NOT the CRUD API) ----
    # These persist the lifecycle the Supervisor drives; they are not gated by
    # `_require_writable` because they record server-internal run progress, not a
    # user config edit.

    def mark_running(self, agent_id: str) -> AgentRecord:
        """Record that a run for this agent has started."""
        if agent_id == ORCHESTRATOR_ID:
            self._orch_state = AgentState.RUNNING
            return self._orchestrator_record()
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
        minted id is appended by ``mark_finished`` (part 1). A backend change
        starts a fresh history (sessions are backend-specific)."""
        self._registry.set_current(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_sessions(self) -> SessionIdList:
        """Every session the registry attributes to the orchestrator under its
        current backend, for the switcher list (part 1). Ownership is recorded,
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
        """Record which agent + orchestrator session spawned ``child_id`` (part 3),
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
        the per-run EventBus has closed (BC1).

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
        # If `request_input` fired during THIS run (BC2), the agent ended its turn
        # deliberately awaiting a decision - the natural DONE that follows must not
        # clobber that needs-input signal. Preserve the same-run, unacknowledged
        # WAITING outcome (and its question), refreshing only the now-finalized
        # session id. Keyed on run_id so a WAITING left by an EARLIER run is still
        # overwritten by a later run's completion, and an ERROR still wins (a crash
        # is not a clean wait). The outcome is built now but WRITTEN only once the
        # agent is known to EXIST (after `_raw` / the orchestrator branch), so a
        # delete-mid-run cannot resurrect it.
        existing = self._outcomes.get(agent_id)
        preserve_waiting = (
            state == AgentState.DONE
            and existing is not None
            and bool(existing.run_id)
            and existing.run_id == run_id
            and existing.state == AgentState.WAITING
            and not existing.acknowledged
        )
        if preserve_waiting:
            assert existing is not None  # narrowed by preserve_waiting
            outcome = existing.model_copy(
                update={
                    "session_id": session_id or existing.session_id,
                    "ts": time.time(),
                }
            )
            eff_state = AgentState.WAITING
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
        if agent_id == ORCHESTRATOR_ID:
            # The orchestrator's run-state is in-memory (it has no agents.json
            # row); only its session id persists, via the registry.
            self._orch_state = eff_state
            if session_id is not None:
                self._registry.set(
                    ORCHESTRATOR_ID, backend or self._orch_backend(), session_id
                )
            self._outcomes.set(agent_id, outcome)
            return self._orchestrator_record()
        agent = self._raw(agent_id)
        if session_id is not None:
            self._registry.set(agent_id, backend or agent.backend, session_id)
        self._outcomes.set(agent_id, outcome)
        updated = agent.model_copy(update={"state": eff_state})
        self._agents[agent_id] = updated
        self._persist()
        return self._with_session(updated)

    def request_input(
        self,
        agent_id: str,
        question: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that a (mid-run) agent is blocked and needs a decision (BC2):
        write a WAITING outcome carrying ``question``, keyed to the current
        ``run_id`` so the turn-end DONE preserves it (see ``mark_finished``).
        Raises AgentNotFound for a missing agent (the caller is a live sub-agent,
        but a delete could race), writing nothing in that case."""
        self._raw(agent_id)  # existence guard; raises before any write
        outcome = RunOutcome(
            state=AgentState.WAITING,
            message=question,
            run_id=run_id,
            session_id=session_id,
            ts=time.time(),
            acknowledged=False,
        )
        self._outcomes.set(agent_id, outcome)
        return outcome

    def outcome(self, agent_id: str) -> RunOutcome | None:
        """The agent's most-recent durable run outcome, or None if it has not
        finished a run yet (BC1)."""
        return self._outcomes.get(agent_id)

    def outcomes(self) -> dict[str, RunOutcome]:
        """All agents' most-recent run outcomes, keyed by agent id. The
        orchestrator's poll ('who needs me', BC3) reads from here."""
        return self._outcomes.all()

    def pending_outcomes(self) -> dict[str, RunOutcome]:
        """The agents that need the orchestrator: those with an UNACKNOWLEDGED
        needs-input (`WAITING`) or `ERROR` outcome (BC3). A cleanly DONE agent is
        not pending; an acknowledged one has been handled."""
        return {
            agent_id: outcome
            for agent_id, outcome in self._outcomes.all().items()
            if not outcome.acknowledged
            and outcome.state in (AgentState.WAITING, AgentState.ERROR)
        }

    def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's outcome handled so it drops out of `pending_outcomes`
        (BC3). Returns True if it flipped an unacknowledged outcome, False if there
        was none or it was already acknowledged (idempotent; never raises for an
        unknown agent - a deleted agent has no outcome to ack)."""
        outcome = self._outcomes.get(agent_id)
        if outcome is None or outcome.acknowledged:
            return False
        self._outcomes.set(agent_id, outcome.model_copy(update={"acknowledged": True}))
        return True
