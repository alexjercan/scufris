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

from .config import (
    Settings,
    available_backends,
    canonical_backend,
    default_model_for,
    normalize_permission_mode,
)
from .projects import ProjectNotFound, ProjectStore

logger = logging.getLogger(__name__)

# An agent id is a path/URL segment (`/api/agents/<id>`), so restrict it to a
# safe charset - no slashes, dots or whitespace (mirrors PROJECT_ID_RE).
AGENT_ID_RE = r"^[A-Za-z0-9_-]+$"

# Lifecycle states an agent moves through; the run machinery (A3) drives them.
AgentLifecycle = Literal["idle", "running", "blocked", "done", "error"]

# An agent's write posture (Claude-style), default manual (read-only).
PermissionMode = Literal["manual", "edit", "auto"]


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
    description: str = ""
    # Retired from the create flow (work is driven by chatting); kept as optional
    # metadata for back-compat with older records.
    goal: str = ""
    task_id: str = ""
    session_id: str | None = None
    state: AgentLifecycle = "idle"
    permission_mode: PermissionMode = "manual"


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
            permission_mode=normalize_permission_mode(permission_mode),  # type: ignore[arg-type]
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
        agent = self.get(agent_id)
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
