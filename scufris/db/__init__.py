"""The app's persistence core: one SQLite database, one transaction boundary.

``engine`` owns the boundary - the factory, the pragma hook and
``Database.transaction()``. ``models.py`` declares the schema and ``migrate``
applies it. ``legacy`` reads an operator's pre-database JSON files into it, once.

:func:`open_state_database` is the startup call that puts those three in the one
order that is correct, and every process that opens the database uses it before
any store reads. The rules a caller has to keep are in ``scufris/db/engine.py``;
where this sits in the app is `scufris/README.md` section 9.
"""

from __future__ import annotations

from pathlib import Path

from .engine import DATABASE_FILENAME, Database, database_path, open_database
from .legacy import LegacyImportRefused, import_legacy_file, import_projects
from .migrate import upgrade_to_head

__all__ = [
    "DATABASE_FILENAME",
    "Database",
    "LegacyImportRefused",
    "database_path",
    "import_legacy_file",
    "import_projects",
    "open_database",
    "open_state_database",
    "upgrade_to_head",
]


def open_state_database(state_dir: Path) -> Database:
    """Open the one database ready for the stores to read, and return it.

    Three steps in the only order that works: open, bring the schema to head,
    then import whatever legacy JSON the operator still has. The import runs
    AFTER the migration because it writes through the models, and BEFORE the
    caller's first store read because a store that read first would report an
    empty database to an operator whose projects are sitting in `projects.json`.

    The handle is long-lived and the CALLER closes it - the stores read through
    it for the process's whole life. On any failure here it is closed before the
    exception leaves, so a refused import does not strand a connection.
    """
    db = open_database(state_dir)
    try:
        upgrade_to_head(db)
        import_projects(db, state_dir)
    except BaseException:
        db.close()
        raise
    return db
