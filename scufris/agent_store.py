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
from typing import Literal

from pydantic import BaseModel

from .config import Settings
from .projects import ProjectNotFound, ProjectStore

logger = logging.getLogger(__name__)

# An agent id is a path/URL segment (`/api/agents/<id>`), so restrict it to a
# safe charset - no slashes, dots or whitespace (mirrors PROJECT_ID_RE).
AGENT_ID_RE = r"^[A-Za-z0-9_-]+$"

# Backends known today; a plain set (not the settings Literal) so extending it
# needs no schema change to persisted records.
KNOWN_BACKENDS: frozenset[str] = frozenset({"app_server", "exec", "mock", "claude"})

# Lifecycle states an agent moves through; the run machinery (A3) drives them.
AgentLifecycle = Literal["idle", "running", "blocked", "done", "error"]


class AgentNotFound(KeyError):
    """Raised when an agent id does not exist."""


class InvalidAgent(ValueError):
    """Raised for an invalid field (empty name, unknown project, bad backend)."""


class AgentsReadOnly(RuntimeError):
    """Raised when a write is attempted while ``settings_writable`` is false."""


class AgentRecord(BaseModel):
    """A configured agent. ``session_id``/``state`` are set by the run machinery,
    not the CRUD API."""

    id: str
    name: str
    project_id: str
    backend: str
    model: str = ""
    goal: str = ""
    task_id: str = ""
    session_id: str | None = None
    state: AgentLifecycle = "idle"
    write_enabled: bool = False


def _slugify(name: str) -> str:
    """A URL-safe slug from a name (mirrors ``projects._slugify``): lowercase,
    non-alnum -> '-', trimmed; non-ASCII dropped; empty -> ``"agent"``. The output
    is provably confined to ``[A-Za-z0-9_-]`` so it can never carry a
    slash/dot/traversal."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug or "agent"


class AgentStore:
    """Owns the persisted list of agents."""

    def __init__(self, settings: Settings, projects: ProjectStore) -> None:
        self._settings = settings
        self._projects = projects
        self._path = Path(settings.state_dir) / "agents.json"
        self._agents: dict[str, AgentRecord] = {}
        self._load()

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
            try:
                agent = AgentRecord.model_validate(item)
            except ValueError as exc:
                logger.warning("agent store: dropping invalid record: %s", exc)
                continue
            self._agents[agent.id] = agent

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [a.model_dump() for a in self._agents.values()]
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def list(self) -> list[AgentRecord]:
        return sorted(self._agents.values(), key=lambda a: a.name.lower())

    def get(self, agent_id: str) -> AgentRecord:
        try:
            return self._agents[agent_id]
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
        goal: str = "",
        task_id: str = "",
        write_enabled: bool = False,
    ) -> AgentRecord:
        self._require_writable()
        name = name.strip()
        if not name:
            raise InvalidAgent("agent name must not be empty")
        try:
            self._projects.get(project_id)
        except ProjectNotFound as exc:
            raise InvalidAgent(f"no such project: {project_id}") from exc
        backend = (backend or self._settings.agent_backend).strip()
        if backend not in KNOWN_BACKENDS:
            raise InvalidAgent(
                f"unknown backend {backend!r}; known: {sorted(KNOWN_BACKENDS)}"
            )
        base = _slugify(name)
        if not re.fullmatch(AGENT_ID_RE, base):
            raise InvalidAgent(f"cannot derive a valid id from name {name!r}")
        agent = AgentRecord(
            id=self._unique_id(base),
            name=name,
            project_id=project_id,
            backend=backend,
            model=(model if model is not None else self._settings.agent_model),
            goal=goal.strip(),
            task_id=task_id.strip(),
            write_enabled=write_enabled,
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
        goal: str | None = None,
        task_id: str | None = None,
        write_enabled: bool | None = None,
    ) -> AgentRecord:
        self._require_writable()
        agent = self.get(agent_id)
        updates: dict[str, object] = {}
        if name is not None:
            name = name.strip()
            if not name:
                raise InvalidAgent("agent name must not be empty")
            updates["name"] = name
        if backend is not None:
            backend = backend.strip()
            if backend not in KNOWN_BACKENDS:
                raise InvalidAgent(
                    f"unknown backend {backend!r}; known: {sorted(KNOWN_BACKENDS)}"
                )
            updates["backend"] = backend
        if model is not None:
            updates["model"] = model
        if goal is not None:
            updates["goal"] = goal.strip()
        if task_id is not None:
            updates["task_id"] = task_id.strip()
        if write_enabled is not None:
            updates["write_enabled"] = write_enabled
        updated = agent.model_copy(update=updates)
        self._agents[agent_id] = updated
        self._persist()
        return updated

    def delete(self, agent_id: str) -> None:
        self._require_writable()
        if agent_id not in self._agents:
            raise AgentNotFound(agent_id)
        del self._agents[agent_id]
        self._persist()

    # --- run-state mutators (used by the run engine, A3; NOT the CRUD API) ----
    # These persist the lifecycle the Supervisor drives; they are not gated by
    # `_require_writable` because they record server-internal run progress, not a
    # user config edit.

    def mark_running(self, agent_id: str) -> AgentRecord:
        """Record that a run for this agent has started."""
        agent = self.get(agent_id)
        updated = agent.model_copy(update={"state": "running"})
        self._agents[agent_id] = updated
        self._persist()
        return updated

    def mark_finished(
        self,
        agent_id: str,
        *,
        state: AgentLifecycle,
        session_id: str | None = None,
    ) -> AgentRecord:
        """Record a run's terminal state and (if produced) its session id."""
        agent = self.get(agent_id)
        updates: dict[str, object] = {"state": state}
        if session_id is not None:
            updates["session_id"] = session_id
        updated = agent.model_copy(update=updates)
        self._agents[agent_id] = updated
        self._persist()
        return updated
