"""The import policy every legacy source goes through: back up, gate, refuse.

This is the mechanism half of ``scufris.db.legacy``; the package docstring is the
policy it implements, and ``loaders.py`` holds what each source does with its
parsed JSON.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Connection, insert, select

from scufris_core import FILE_MODE, Database

from ..models import LegacyImportRow

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".pre-sqlite.bak"

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


def import_legacy_file(
    db: Database, source: Path, load: Loader, *, key: str | None = None
) -> bool:
    """Import one legacy JSON file into ``db``, at most once.

    Returns True if this call imported it, False if a previous run already did
    or the file does not exist (a fresh install has none). Raises
    ``LegacyImportRefused`` for a file that exists and cannot be trusted; the
    database is unchanged and the gate stays open, so a repaired file imports on
    the next run.

    ``key`` is what the ``legacy_import`` row records, defaulting to the file's
    name. The reasoning sidecar needs it explicitly: its files are named after a
    SESSION ID, and a session called ``sessions`` would produce ``sessions.json``
    and collide with the session registry's own gate row - which would silently
    skip whichever ran second. The sidecar passes ``reasoning/<name>``.

    Everything happens inside ONE transaction, including the backup and the read
    - so the check that the source has not already been imported and the writing
    of its ``legacy_import`` row cannot be separated by another process. The
    write lock is therefore held across a small file read, which is affordable
    because this runs once, at startup, on files a single host wrote by hand.
    """
    gate = key if key is not None else source.name
    with db.transaction() as conn:
        if _is_imported(conn, gate):
            return False
        if not source.is_file():
            return False
        backup = _back_up(source)
        count = load(source, conn, _parse(source))
        conn.execute(insert(LegacyImportRow).values(source=gate, imported_at=_now()))
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
