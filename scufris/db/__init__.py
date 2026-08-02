"""The app's persistence core: one SQLite database, one transaction boundary.

``engine`` owns the boundary - the factory, the pragma hook and
``Database.transaction()``. ``models.py`` declares the schema and ``migrate``
applies it; every process that opens the database calls
``migrate_state_dir(state_dir)`` at startup, before any store reads it.
``legacy`` reads an operator's pre-database JSON files into it, once - nothing
calls it until the store cutover. The rules a caller has to keep are in
``scufris/db/engine.py``; where this sits in the app is `scufris/README.md`
section 9.
"""

from __future__ import annotations

from .engine import DATABASE_FILENAME, Database, database_path, open_database
from .legacy import LegacyImportRefused, import_legacy_file, import_projects
from .migrate import migrate_state_dir, upgrade_to_head

__all__ = [
    "DATABASE_FILENAME",
    "Database",
    "LegacyImportRefused",
    "database_path",
    "import_legacy_file",
    "import_projects",
    "migrate_state_dir",
    "open_database",
    "upgrade_to_head",
]
