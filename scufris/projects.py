"""First-class projects: a workspace record persisted to the state database.

A project is `{id, cwd, name, language, description}` - the organizing unit for
the projects-orchestrator concept. This module owns only the STORE + its
records; per-project agents, skills and tools are later phases.

The store reads THROUGH the database: every method opens one
``Database.transaction()``, and there is no in-memory mirror to go stale or to
publish a write that failed. 20260729-102146 measured the cost of the mirror the
JSON store had - 97 of 97 failed writes stayed live in the process and were
published by the next successful one.

What crosses this boundary is always a :class:`Project`, never a row object: the
FastAPI routes serialize what these methods return, and a `Session`-bound
instance in a response is a detached-load failure waiting for the first lazy
attribute. Nothing here uses the ORM session at all - the statements run on the
Core ``Connection`` the transaction yields.

Writes are still gated by ``settings_writable``.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Connection, Row, insert, select
from sqlalchemy import delete as sql_delete
from sqlalchemy import update as sql_update
from sqlalchemy.exc import IntegrityError

from .config import Settings
from .db import Database
from .db.models import ProjectRow

logger = logging.getLogger(__name__)

# Wall-clock cap for a `tatr ls` shell-out (never blocks the endpoint).
_TATR_TIMEOUT = 10.0
# One `tatr ls` line: "<path>: [PRIORITY: N, ..., TAGS: a, b] Title".
#
# The fields BETWEEN priority and tags are skipped rather than named: tatr has
# already added `KIND` and `FLOW STEP` there since this was written, and pinning
# the exact field list is what made every task silently disappear from the
# Projects page instead of failing loudly. Only the two fields this module
# actually reads are matched.
_TASK_LINE_RE = re.compile(
    r"^(?P<path>.+?): \[PRIORITY: (?P<pri>-?\d+), (?:[^\]]*?, )?"
    r"TAGS: (?P<tags>[^\]]*)\] (?P<title>.*)$"
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


def _record(row: Row[Any]) -> Project:
    """The pydantic record for one selected row. Nothing else leaves the store."""
    return Project.model_validate(dict(row._mapping))


def _fetch(conn: Connection, project_id: str) -> Project | None:
    row = conn.execute(
        select(ProjectRow.__table__).where(ProjectRow.id == project_id)
    ).first()
    return None if row is None else _record(row)


def _unique_id(conn: Connection, base: str) -> str:
    """``base``, or the first ``base-N`` free of it.

    Called INSIDE the caller's transaction, which is what closes the
    read-modify-write window: every begin on this engine is immediate, so no
    other writer - in this process or another - can take the id between the
    lookup here and the insert that follows.
    """
    taken = set(
        conn.scalars(
            # autoescape because a slug may contain '_', which is a LIKE
            # wildcard: unescaped, `a_b` would also match `axb` and inflate the
            # taken set.
            select(ProjectRow.id).where(ProjectRow.id.startswith(base, autoescape=True))
        ).all()
    )
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


class ProjectStore:
    """Owns the persisted list of projects, in the state database."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _require_writable(self) -> None:
        if not self.writable:
            raise ProjectsReadOnly("projects are read-only on this server")

    def list(self) -> list[Project]:
        with self._db.transaction() as conn:
            rows = conn.execute(select(ProjectRow.__table__)).all()
        # Ordered here rather than in SQL: SQLite's own lower() is ASCII-only,
        # and the ordering the API has always published is Python's.
        return sorted((_record(row) for row in rows), key=lambda p: p.name.lower())

    def get(self, project_id: str) -> Project:
        with self._db.transaction() as conn:
            project = _fetch(conn, project_id)
        if project is None:
            raise ProjectNotFound(project_id)
        return project

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
        # Validated before the transaction: `is_dir` is a filesystem call, and
        # the write lock is held by every begin on this engine.
        resolved = Path(cwd).expanduser()
        if not resolved.is_dir():
            raise InvalidProject(f"cwd is not an existing directory: {cwd}")
        base = _slugify(name)
        if not re.fullmatch(PROJECT_ID_RE, base):
            raise InvalidProject(f"cannot derive a valid id from name {name!r}")
        with self._db.transaction() as conn:
            project = Project(
                id=_unique_id(conn, base),
                cwd=str(resolved),
                name=name,
                language=language.strip(),
                description=description.strip(),
            )
            try:
                conn.execute(insert(ProjectRow).values(**project.model_dump()))
            except IntegrityError as exc:
                # The id is a real PRIMARY KEY now. Dedup above shares this
                # transaction so nothing outside can win the race, but the
                # constraint is the authority and its violation is a domain
                # error, not a database error surfacing at a route.
                raise DuplicateProject(project.id) from exc
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
        updates: dict[str, str] = {}
        # The record is resolved BEFORE the fields are validated, the order the
        # JSON store had: an unknown id is not-found whatever else is wrong with
        # the request, and the routes map the two to 404 and 422, so the order
        # they run in is the status an operator sees. That puts `is_dir` - a
        # filesystem call - inside the transaction, where `create` keeps it
        # outside: one stat while the write lock is held is far cheaper than the
        # second immediate begin an existence probe ahead of it would cost.
        with self._db.transaction() as conn:
            current = _fetch(conn, project_id)
            if current is None:
                raise ProjectNotFound(project_id)
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
            if updates:
                conn.execute(
                    sql_update(ProjectRow)
                    .where(ProjectRow.id == project_id)
                    .values(**updates)
                )
        return current.model_copy(update=updates)

    def delete(self, project_id: str) -> None:
        self._require_writable()
        with self._db.transaction() as conn:
            result = conn.execute(
                sql_delete(ProjectRow).where(ProjectRow.id == project_id)
            )
            if result.rowcount == 0:
                raise ProjectNotFound(project_id)


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
