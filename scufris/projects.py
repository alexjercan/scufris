"""First-class projects: a workspace record persisted to a state file.

A project is `{id, cwd, name, language, description}` - the organizing unit for
the projects-orchestrator concept. This module owns only the STORE + its
records; per-project agents, skills and tools are later phases. Persistence
mirrors the settings store: one JSON file under the state dir, atomic write,
tolerant load, writes gated by ``settings_writable``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)

# Wall-clock cap for a `tatr ls` shell-out (never blocks the endpoint).
_TATR_TIMEOUT = 10.0
# One `tatr ls` line: "<path>: [PRIORITY: N, TAGS: a, b] Title".
_TASK_LINE_RE = re.compile(
    r"^(?P<path>.+?): \[PRIORITY: (?P<pri>-?\d+), TAGS: (?P<tags>[^\]]*)\] "
    r"(?P<title>.*)$"
)

# A project id is a path/URL segment (`/api/projects/<id>`), so restrict it to a
# safe charset - no slashes, dots or whitespace.
PROJECT_ID_RE = r"^[A-Za-z0-9_-]+$"


class ProjectNotFound(KeyError):
    """Raised when a project id does not exist."""


class InvalidProject(ValueError):
    """Raised for an invalid field (empty name, missing cwd)."""


class DuplicateProject(ValueError):
    """Raised when a create would collide with an existing id after dedup."""


class ProjectsReadOnly(RuntimeError):
    """Raised when a write is attempted while ``settings_writable`` is false."""


class Project(BaseModel):
    id: str
    cwd: str
    name: str
    language: str = ""
    description: str = ""


def _slugify(name: str) -> str:
    """A URL-safe slug from a name: lowercase, non-alnum -> '-', trimmed.

    Non-ASCII is dropped, not transliterated (``Cafe`` and ``Caf' e`` can slug
    to the same base) - intentional: dedup gives distinct ids, and the id is a
    URL segment where a confined `[A-Za-z0-9_-]` charset matters more than
    fidelity. The output is provably confined to that charset (empty ->
    ``"project"``), so it can never carry a slash/dot/traversal.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug or "project"


class ProjectStore:
    """Owns the persisted list of projects."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = Path(settings.state_dir) / "projects.json"
        self._projects: dict[str, Project] = {}
        self._load()

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _require_writable(self) -> None:
        if not self.writable:
            raise ProjectsReadOnly("projects are read-only on this server")

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("project store: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, list):
            return
        for item in data:
            try:
                project = Project.model_validate(item)
            except ValueError as exc:
                logger.warning("project store: dropping invalid record: %s", exc)
                continue
            self._projects[project.id] = project

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [p.model_dump() for p in self._projects.values()]
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def list(self) -> list[Project]:
        return sorted(self._projects.values(), key=lambda p: p.name.lower())

    def get(self, project_id: str) -> Project:
        try:
            return self._projects[project_id]
        except KeyError as exc:
            raise ProjectNotFound(project_id) from exc

    def _unique_id(self, base: str) -> str:
        if base not in self._projects:
            return base
        n = 2
        while f"{base}-{n}" in self._projects:
            n += 1
        return f"{base}-{n}"

    def create(
        self,
        name: str,
        cwd: str,
        language: str = "",
        description: str = "",
    ) -> Project:
        self._require_writable()
        name = name.strip()
        if not name:
            raise InvalidProject("project name must not be empty")
        resolved = Path(cwd).expanduser()
        if not resolved.is_dir():
            raise InvalidProject(f"cwd is not an existing directory: {cwd}")
        base = _slugify(name)
        if not re.fullmatch(PROJECT_ID_RE, base):
            raise InvalidProject(f"cannot derive a valid id from name {name!r}")
        project = Project(
            id=self._unique_id(base),
            cwd=str(resolved),
            name=name,
            language=language.strip(),
            description=description.strip(),
        )
        self._projects[project.id] = project
        self._persist()
        return project

    def update(
        self,
        project_id: str,
        *,
        name: str | None = None,
        language: str | None = None,
        description: str | None = None,
        cwd: str | None = None,
    ) -> Project:
        self._require_writable()
        project = self.get(project_id)
        updates: dict[str, str] = {}
        if name is not None:
            name = name.strip()
            if not name:
                raise InvalidProject("project name must not be empty")
            updates["name"] = name
        if cwd is not None:
            resolved = Path(cwd).expanduser()
            if not resolved.is_dir():
                raise InvalidProject(f"cwd is not an existing directory: {cwd}")
            updates["cwd"] = str(resolved)
        if language is not None:
            updates["language"] = language.strip()
        if description is not None:
            updates["description"] = description.strip()
        updated = project.model_copy(update=updates)
        self._projects[project_id] = updated
        self._persist()
        return updated

    def delete(self, project_id: str) -> None:
        self._require_writable()
        if project_id not in self._projects:
            raise ProjectNotFound(project_id)
        del self._projects[project_id]
        self._persist()


class ProjectTask(BaseModel):
    """One tatr task belonging to a project (the specs in spec-driven dev)."""

    id: str
    title: str
    priority: int
    tags: list[str] = []


def read_project_tasks(cwd: str) -> list[ProjectTask]:
    """The tatr tasks under a project's cwd, parsed into records.

    Scoped to ``<cwd>/tasks``: if that directory does not exist we return an
    empty list WITHOUT calling tatr, so tatr cannot walk UP to a parent's
    ``tasks/`` (its `-r` searches upward). Never raises - a missing tatr, a
    timeout or a non-zero exit yields an empty list and a log line.
    """
    root = Path(cwd)
    if not (root / "tasks").is_dir():
        return []
    exe = shutil.which("tatr")
    if exe is None:
        logger.info("read_project_tasks: tatr not on PATH")
        return []
    try:
        proc = subprocess.run(
            [exe, "-r", str(root), "ls"],
            capture_output=True,
            text=True,
            timeout=_TATR_TIMEOUT,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("read_project_tasks: tatr failed: %s", exc)
        return []
    if proc.returncode != 0:
        logger.info("read_project_tasks: tatr exit=%d", proc.returncode)
        return []
    tasks: list[ProjectTask] = []
    for line in proc.stdout.splitlines():
        match = _TASK_LINE_RE.match(line.strip())
        if match is None:
            continue
        # The task id is its directory name (…/tasks/<id>/TASK.md).
        task_id = Path(match.group("path")).parent.name
        tags = [t.strip() for t in match.group("tags").split(",") if t.strip()]
        tasks.append(
            ProjectTask(
                id=task_id,
                title=match.group("title").strip(),
                priority=int(match.group("pri")),
                tags=tags,
            )
        )
    return tasks
