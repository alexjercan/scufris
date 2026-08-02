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


def database(settings: "Settings") -> "Database":
    """This process's handle on the state database, opened and prepared once.

    The memo itself lives in ``scufris.db`` now, because the dashboard reaches it
    too: ``create_app`` and this subprocess have to agree on ONE handle per
    process, and a second map here would have made "one handle" true of each
    module rather than of the process. Nothing in an MCP subprocess closes it -
    the process exiting is what releases the file.
    """
    from .db import state_database

    return state_database(Path(settings.state_dir))


def project_store(settings: "Settings") -> "ProjectStore":
    from .projects import ProjectStore

    return ProjectStore(settings, database(settings))


def agent_store(settings: "Settings") -> "AgentStore":
    from .agent_store import AgentStore

    return AgentStore(settings, project_store(settings), database(settings))
