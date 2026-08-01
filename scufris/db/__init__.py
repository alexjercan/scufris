"""The app's persistence core: one SQLite database, one transaction boundary.

``engine`` is the whole public surface today - the factory, the pragma hook and
``Database.transaction()``. ``models.py``, ``migrations/`` and ``legacy.py``
arrive in the following tasks. The rules a caller has to keep are in
``scufris/db/engine.py``; where this sits in the app is `scufris/README.md`
section 9.
"""

from __future__ import annotations

from .engine import DATABASE_FILENAME, Database, database_path, open_database

__all__ = [
    "DATABASE_FILENAME",
    "Database",
    "database_path",
    "open_database",
]
