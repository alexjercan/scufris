"""How an MCP subprocess reaches the app's persisted state.

An MCP server runs in its own process, spawned by a backend. It does not share
the dashboard's objects, so the tools that observe agents read the same files the
dashboard writes - which since the store cutover means the same SQLite database.

This module owns that handle and the stores built on it, so each server's tool
module holds only its tools. Imports are deferred into the functions: the servers
are started for their tools, and a `den` server on a box with no state database
should not pay for SQLAlchemy at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_store import AgentStore
    from .config import Settings
    from .db import Database
    from .projects import ProjectStore

# This process's open state databases, keyed by the RESOLVED state directory.
# One entry in production - the process serves one state dir for its whole life,
# and re-opening per tool call would re-run the pragma hook and the WAL handshake
# on every observation. Keyed rather than a single slot because tests drive these
# against a fresh temp state dir each time. Nothing closes them: the process
# exiting is what releases the file.
_DATABASES: dict[Path, "Database"] = {}


def database(settings: "Settings") -> "Database":
    """This process's handle on the state database, opened and prepared once.

    The full startup sequence runs here rather than being assumed: a backend can
    spawn this subprocess on a machine where the dashboard is not the process
    that ran first.
    """
    from .db import open_state_database

    key = Path(settings.state_dir).resolve()
    db = _DATABASES.get(key)
    if db is None:
        db = open_state_database(key)
        _DATABASES[key] = db
    return db


def project_store(settings: "Settings") -> "ProjectStore":
    from .projects import ProjectStore

    return ProjectStore(settings, database(settings))


def agent_store(settings: "Settings") -> "AgentStore":
    from .agent_store import AgentStore

    return AgentStore(settings, project_store(settings))
