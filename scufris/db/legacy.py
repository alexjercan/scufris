"""Read an operator's legacy per-store JSON files into the database, once.

The policy lives here rather than in each store's importer:

- **Backed up first.** The source is copied to ``<name>.pre-sqlite.bak`` before
  it is read, at 0600 from creation - the later store migrations bring auth
  sessions through this same function.
- **Never deleted.** Nothing here removes a legacy file. The operator does that,
  once they are satisfied, and that deletion is what makes the move one-way.
- **Damaged is refused, never treated as empty.** A file that does not parse is
  named with its line, column and the parser's own message; a record that fails
  its pydantic model fails the WHOLE import. There is no tolerant loader on this
  path: the JSON store this replaces logged and skipped such a record, and
  importing under that rule would leave a database quietly missing records the
  operator can still see in their JSON.
- **All or nothing, once.** One source is imported inside one
  ``Database.transaction()`` that also writes its ``legacy_import`` row, so a
  failure anywhere leaves no rows AND no gate: the operator repairs the file and
  the retry starts from the beginning.

The gate is a table rather than a schema version, and the import cannot ride a
schema revision: it needs the state directory and the pydantic models to do its
job, and neither belongs inside a migration.

``scufris.db.open_state_database`` is the only call site: it runs the import at
startup, after the migration and ahead of the first store read. Both orderings
matter - importing before the migration would write through models the schema
does not have yet, and importing after a store had already read would show an
operator an empty database while their projects sat in ``projects.json``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import Connection, insert, select

from .engine import FILE_MODE, Database
from .models import LegacyImportRow, ProjectRow

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".pre-sqlite.bak"

PROJECTS_FILENAME = "projects.json"

# What one store's importer does with its parsed JSON: validate it, write it on
# the OPEN connection, and return how many records it wrote. It never opens a
# transaction of its own - the one it is called inside is what makes the import
# all-or-nothing - and it raises `LegacyImportRefused` rather than dropping a
# record it cannot validate.
Loader = Callable[[Path, Connection, object], int]


class LegacyImportRefused(RuntimeError):
    """The legacy file cannot be imported, and is NOT to be treated as absent."""


def backup_path(source: Path) -> Path:
    """Where ``source`` is copied before it is read."""
    return source.with_name(f"{source.name}{BACKUP_SUFFIX}")


def import_projects(db: Database, state_dir: Path) -> bool:
    """Import ``projects.json`` from ``state_dir``. True if this run imported it."""
    return import_legacy_file(db, Path(state_dir) / PROJECTS_FILENAME, _load_projects)


def import_legacy_file(db: Database, source: Path, load: Loader) -> bool:
    """Import one legacy JSON file into ``db``, at most once.

    Returns True if this call imported it, False if a previous run already did
    or the file does not exist (a fresh install has none). Raises
    ``LegacyImportRefused`` for a file that exists and cannot be trusted; the
    database is unchanged and the gate stays open, so a repaired file imports on
    the next run.

    Everything happens inside ONE transaction, including the backup and the read
    - so the check that the source has not already been imported and the writing
    of its ``legacy_import`` row cannot be separated by another process. The
    write lock is therefore held across a small file read, which is affordable
    because this runs once, at startup, on files a single host wrote by hand.
    """
    with db.transaction() as conn:
        if _is_imported(conn, source.name):
            return False
        if not source.is_file():
            return False
        backup = _back_up(source)
        count = load(source, conn, _parse(source))
        conn.execute(
            insert(LegacyImportRow).values(source=source.name, imported_at=_now())
        )
    logger.info(
        "imported %d records from %s (the file is left in place; backup: %s)",
        count,
        source,
        backup,
    )
    return True


def _is_imported(conn: Connection, name: str) -> bool:
    row = conn.execute(
        select(LegacyImportRow.source).where(LegacyImportRow.source == name)
    ).first()
    return row is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _back_up(source: Path) -> Path:
    """Copy ``source`` next to itself at 0600, replacing an earlier attempt's copy.

    Created 0600 rather than chmod-ed to it afterwards, and never through a
    symlink: this function is how an operator's auth sessions will reach disk a
    second time, and a window where that copy is world-readable is the same
    exposure whether it lasts a millisecond or a day.

    A leftover backup is a previous import that failed after the copy. Nothing
    here ever writes to the source, so the file has not moved since and the
    fresh copy is the same bytes.
    """
    target = backup_path(source)
    if target.is_symlink():
        raise LegacyImportRefused(
            f"REFUSED: {target} is a symlink; refusing to write the backup"
        )
    target.unlink(missing_ok=True)
    fd = os.open(
        target,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        FILE_MODE,
    )
    with os.fdopen(fd, "wb") as handle:
        handle.write(source.read_bytes())
    return target


def _parse(source: Path) -> object:
    """The file's JSON, or a refusal naming where it stops making sense."""
    try:
        text = source.read_text()
    except OSError as exc:
        raise LegacyImportRefused(f"REFUSED: {source} cannot be read: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LegacyImportRefused(
            f"REFUSED: {source} is damaged at line {exc.lineno} col {exc.colno}: "
            f"{exc.msg}. {backup_path(source)} is a copy of this same damaged "
            "file, not a repair - restoring it changes nothing. Repair the file "
            "from your own backup, or move it aside to import the rest of the "
            "state without it."
        ) from exc


def _load_projects(source: Path, conn: Connection, payload: object) -> int:
    """Validate every record through :class:`scufris.projects.Project` and write it.

    Inserted one at a time as they validate, so the rollback that an invalid
    record triggers is what removes the records BEFORE it. Validating the whole
    file up front would pass the same test without the transaction doing
    anything.

    ``Project`` is imported HERE rather than at module scope: since the store
    cutover, ``scufris.projects`` imports this package for its ``Database`` and
    ``ProjectRow``, and a top-level import back into it makes the two modules
    load-order dependent - importing ``scufris.projects`` first would reach this
    line before ``Project`` exists.
    """
    from ..projects import Project

    if not isinstance(payload, list):
        raise LegacyImportRefused(
            f"REFUSED: {source} is damaged: the top level is "
            f"{type(payload).__name__}, not a list of projects"
        )
    count = 0
    for index, item in enumerate(payload):
        try:
            project = Project.model_validate(item)
        except ValidationError as exc:
            raise LegacyImportRefused(
                f"REFUSED: {source} record {index} is not a valid project: {exc}"
            ) from exc
        conn.execute(insert(ProjectRow).values(**project.model_dump()))
        count += 1
    return count
