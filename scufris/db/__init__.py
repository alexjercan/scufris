"""The app's persistence core: one SQLite database, one transaction boundary.

``engine`` owns the boundary - the factory, the pragma hook and
``Database.transaction()``. ``models.py`` declares the schema and ``migrate``
applies it. ``legacy`` reads an operator's pre-database JSON files into it, once.

:func:`open_state_database` is the startup call that puts those three in the one
order that is correct, and every process that opens the database uses it before
any store reads. :func:`state_database` is how a caller that CANNOT be handed the
handle reaches the one this process already opened. The rules a caller has to
keep are in ``scufris/db/engine.py``; where this sits in the app is
`scufris/README.md` section 9.
"""

from __future__ import annotations

from pathlib import Path

from .engine import DATABASE_FILENAME, Database, database_path, open_database
from .legacy import (
    LegacyImportRefused,
    import_agent_state,
    import_legacy_file,
    import_projects,
)
from .migrate import upgrade_to_head

__all__ = [
    "DATABASE_FILENAME",
    "Database",
    "LegacyImportRefused",
    "close_all_state_databases",
    "close_state_database",
    "database_path",
    "import_agent_state",
    "import_legacy_file",
    "import_projects",
    "open_database",
    "open_state_database",
    "state_database",
    "upgrade_to_head",
]

# This process's open state databases, keyed by the RESOLVED state directory.
# ONE entry in production - a process serves one state dir for its whole life.
# Keyed rather than a single slot because tests drive this against a fresh temp
# state dir each time, and because an MCP subprocess and the dashboard are
# different processes with different maps.
_HANDLES: dict[Path, Database] = {}


def state_database(state_dir: Path) -> Database:
    """This process's handle on ``state_dir``'s database, opened once.

    The one handle per process is the epic's premise: two handles on one file are
    two connection pools contending on one write lock, and the nesting guard -
    which keys on the resolved path - would then be the only thing standing
    between them and a deadlock it can only report, not prevent.

    Injection is still the preferred way in, and a caller that COULD take a
    ``Database`` parameter and reaches for this instead is a review finding.
    This exists for the two places where injection is not available: an MCP
    subprocess, which shares no objects with the dashboard, and
    ``CodexBackend.read_transcript``, whose ``AgentBackend`` protocol passes no
    handles and whose whole point is that nothing above it branches on backend
    (DECISION.md 3 of 20260801-100409).

    The full startup sequence runs here rather than being assumed: a backend can
    spawn an MCP subprocess on a machine where the dashboard is not the process
    that ran first.
    """
    key = Path(state_dir).resolve()
    db = _HANDLES.get(key)
    if db is None:
        db = open_state_database(key)
        _HANDLES[key] = db
    return db


def close_state_database(state_dir: Path) -> None:
    """Close this process's handle on ``state_dir`` and forget it.

    EVICTS as well as closes: a closed engine still answers ``transaction()`` by
    dialling a fresh connection, so leaving the entry behind would hand the next
    caller a handle whose lifespan has already ended - and in tests, one pointing
    at a temp directory that no longer exists.

    A no-op when this process never opened that directory.
    """
    db = _HANDLES.pop(Path(state_dir).resolve(), None)
    if db is not None:
        db.close()


def close_all_state_databases() -> None:
    """Close and forget every handle this process opened through the accessor.

    Production closes exactly one, by name, in the app's lifespan. This exists
    for the TEST process, which opens a handle per temp state directory and
    would otherwise carry them - and their pooled connections - into every later
    test, with a memo pointing at directories pytest has already removed.
    """
    for db in list(_HANDLES.values()):
        db.close()
    _HANDLES.clear()


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
        import_agent_state(db, state_dir)
    except BaseException:
        db.close()
        raise
    return db
