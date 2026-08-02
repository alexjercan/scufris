"""The legacy JSON import: once, backed up, all-or-nothing, never tolerant.

Each proof here discriminates against the behaviour the store cutover is
REMOVING rather than restating the happy path. `ProjectStore._load` logs and
skips an invalid record and returns silently on a damaged file; an import that
did either would leave an operator with a database quietly missing the records
they still see in their JSON, which is the failure this epic exists to end.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from sqlalchemy import select

from scufris.db import Database, open_database, upgrade_to_head
from scufris.db.legacy import (
    BACKUP_SUFFIX,
    LegacyImportRefused,
    backup_path,
    import_projects,
)
from scufris.db.models import LegacyImportRow, ProjectRow

RECORDS = [
    {"id": "scufris", "cwd": "/tmp", "name": "Scufris", "language": "python"},
    {"id": "den", "cwd": "/tmp", "name": "Den", "description": "the journal"},
]


def _write(state_dir: Path, payload: object) -> Path:
    source = state_dir / "projects.json"
    source.write_text(json.dumps(payload, indent=2))
    return source


def _write_text(state_dir: Path, text: str) -> Path:
    source = state_dir / "projects.json"
    source.write_text(text)
    return source


def _ids(db: Database) -> list[str]:
    with db.transaction() as conn:
        return list(
            conn.execute(select(ProjectRow.id).order_by(ProjectRow.id)).scalars()
        )


def _completed(db: Database) -> list[str]:
    with db.transaction() as conn:
        return list(conn.execute(select(LegacyImportRow.source)).scalars())


@pytest.fixture
def make_state(tmp_path: Path) -> Iterator[Callable[[str], tuple[Path, Database]]]:
    """A second (third, ...) migrated state directory, each with its own database.

    The gate is keyed by the source file's NAME, so a test that needs a source
    file imported for the first time twice needs two databases, not two
    directories.
    """
    with ExitStack() as stack:

        def make(name: str) -> tuple[Path, Database]:
            state_dir = tmp_path / name
            state_dir.mkdir()
            db = open_database(state_dir)
            stack.callback(db.close)
            upgrade_to_head(db)
            return state_dir, db

        yield make


def test_legacy_projects_import_is_idempotent_and_refuses_damage(
    database: Database,
    tmp_path: Path,
    make_state: Callable[[str], tuple[Path, Database]],
) -> None:
    """One import, a second run that does nothing, and damage that is refused.

    The idempotency half fails without the `legacy_import` gate (four rows, or
    an IntegrityError, on the second run). The damage half fails against a
    tolerant loader, which would report success having imported nothing.
    """
    _write(tmp_path, RECORDS)

    assert import_projects(database, tmp_path) is True
    assert _ids(database) == ["den", "scufris"]
    assert _completed(database) == ["projects.json"]

    assert import_projects(database, tmp_path) is False
    assert _ids(database) == ["den", "scufris"]

    damaged_dir, damaged_db = make_state("damaged")
    source = _write_text(damaged_dir, json.dumps(RECORDS) + "\n{trailing}")

    with pytest.raises(LegacyImportRefused) as excinfo:
        import_projects(damaged_db, damaged_dir)

    message = str(excinfo.value)
    assert str(source) in message
    assert "line 2 col 1" in message
    assert "Extra data" in message
    # The `.bak` beside a REFUSED file is a copy of the damage this run just
    # took, so the remedy must not send the operator back to it: restoring it is
    # a no-op at best, and overwrites their own repair at worst.
    assert "not a repair" in message
    assert "move it aside" in message
    # Refused, not half-done: nothing imported and the gate is NOT closed, so an
    # operator who repairs the file gets a retry rather than a skipped store.
    assert _ids(damaged_db) == []
    assert _completed(damaged_db) == []
    # The backup is taken BEFORE the file is read, so it survives the refusal.
    assert backup_path(source).read_text() == source.read_text()


def test_legacy_import_backs_up_and_never_deletes_the_source(
    database: Database, tmp_path: Path
) -> None:
    """The source is copied to `<name>.pre-sqlite.bak` and left exactly as it was."""
    source = _write(tmp_path, RECORDS)
    original = source.read_text()

    assert import_projects(database, tmp_path) is True

    assert source.is_file(), "the import deleted the legacy file"
    assert source.read_text() == original, "the import rewrote the legacy file"

    backup = tmp_path / f"projects.json{BACKUP_SUFFIX}"
    assert backup.read_text() == original
    # 0600 from creation, like the database and its own backups: the later store
    # migrations copy auth sessions through this same function.
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_legacy_import_rolls_back_on_an_invalid_record(
    database: Database, tmp_path: Path
) -> None:
    """One invalid record fails the whole import and leaves no rows behind.

    Two discriminations in one: the record BEFORE the invalid one is inserted
    and then rolled back, so dropping the transaction fails this; and skipping
    the invalid record - what `ProjectStore._load` does today - fails it too.
    """
    invalid = {"id": "broken", "name": "No cwd"}
    source = _write(tmp_path, [RECORDS[0], invalid])

    with pytest.raises(LegacyImportRefused) as excinfo:
        import_projects(database, tmp_path)

    message = str(excinfo.value)
    assert str(source) in message
    assert "cwd" in message

    assert _ids(database) == []
    assert _completed(database) == []

    # The operator repairs the file; the retry imports the whole store.
    _write(tmp_path, RECORDS)
    assert import_projects(database, tmp_path) is True
    assert _ids(database) == ["den", "scufris"]


def test_a_symlinked_backup_target_is_refused(
    database: Database, tmp_path: Path
) -> None:
    """The backup is never written THROUGH a symlink, as elsewhere in this package.

    Refused as `LegacyImportRefused`, like every other legacy file that cannot be
    trusted, rather than as a bare `RuntimeError`: the cutover's call site
    catches the documented exception, and this condition means the same to it.
    """
    source = _write(tmp_path, RECORDS)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched")
    backup_path(source).symlink_to(elsewhere)

    with pytest.raises(LegacyImportRefused, match="symlink"):
        import_projects(database, tmp_path)

    assert elsewhere.read_text() == "untouched"
    assert _ids(database) == []


def test_an_absent_legacy_file_is_not_an_import(
    database: Database, tmp_path: Path
) -> None:
    """A fresh install has no legacy file, and that is not a failure or an import."""
    assert import_projects(database, tmp_path) is False
    assert _completed(database) == []
