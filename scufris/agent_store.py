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
from pathlib import Path

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
    """The persisted `(agent_id -> current backend session)` mapping - the ONLY
    home of session ids, for ALL agents (the orchestrator included).

    Each entry records the backend the id belongs to, because a session id is
    backend-specific (a codex rollout id means nothing to claude - task
    20260721-152034): `get` returns None on a backend mismatch, so a stale
    cross-backend id is structurally unreachable. Persistence mirrors the other
    stores (one JSON file under the state dir, atomic write, tolerant load).
    Not gated by ``settings_writable``: like the run-state mutators, it records
    server-internal run progress, not a user config edit."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.state_dir) / "sessions.json"
        self._sessions: dict[str, dict[str, str]] = {}
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
            session_id = entry.get("session_id")
            if isinstance(backend, str) and isinstance(session_id, str):
                self._sessions[agent_id] = {
                    "backend": backend,
                    "session_id": session_id,
                }

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._sessions, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def get(self, agent_id: str, backend: str) -> str | None:
        """The agent's current session id under ``backend``, or None when there
        is no mapping or the stored id belongs to another backend."""
        entry = self._sessions.get(agent_id)
        if entry is None or entry["backend"] != backend:
            return None
        return entry["session_id"]

    def has(self, agent_id: str) -> bool:
        """Whether ANY mapping exists for this agent (backend-agnostic; the
        legacy-migration guard)."""
        return agent_id in self._sessions

    def set(self, agent_id: str, backend: str, session_id: str) -> None:
        self._sessions[agent_id] = {"backend": backend, "session_id": session_id}
        self._persist()

    def clear(self, agent_id: str) -> None:
        if self._sessions.pop(agent_id, None) is not None:
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
        # Drop the session mapping with the agent, so a future agent that
        # happens to reuse the freed id can never inherit this conversation.
        self._registry.clear(agent_id)
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
        """Set (switch to) or clear (start a fresh conversation) the
        orchestrator's active session, in the persisted registry (keyed by the
        current settings backend). Multi-session lives here now that the
        orchestrator runs through the unified backend path (B5bc)."""
        if session_id is None:
            self._registry.clear(ORCHESTRATOR_ID)
        else:
            self._registry.set(ORCHESTRATOR_ID, self._orch_backend(), session_id)

    def orchestrator_session_id(self) -> str | None:
        """The orchestrator's current active session id (or None for fresh).
        Registry-backed: it survives a restart, and a stale id recorded under a
        different backend reads as None instead of being resumed."""
        return self._registry.get(ORCHESTRATOR_ID, self._orch_backend())

    def mark_finished(
        self,
        agent_id: str,
        *,
        state: AgentState,
        session_id: str | None = None,
    ) -> AgentRecord:
        """Record a run's terminal state and (if produced) its session id. The
        session id goes to the registry - for EVERY agent, orchestrator
        included - keyed by the agent's current backend."""
        # Coerce a raw string to the enum: `model_copy(update=...)` below does NOT
        # validate, so a str here would settle on the AgentState field unconverted
        # and later trip pydantic's enum serializer.
        state = AgentState(state)
        if agent_id == ORCHESTRATOR_ID:
            # The orchestrator's run-state is in-memory (it has no agents.json
            # row); only its session id persists, via the registry.
            self._orch_state = state
            if session_id is not None:
                self._registry.set(ORCHESTRATOR_ID, self._orch_backend(), session_id)
            return self._orchestrator_record()
        agent = self._raw(agent_id)
        if session_id is not None:
            self._registry.set(agent_id, agent.backend, session_id)
        updated = agent.model_copy(update={"state": state})
        self._agents[agent_id] = updated
        self._persist()
        return self._with_session(updated)
