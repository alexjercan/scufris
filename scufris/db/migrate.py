"""Bring the state database up to the schema this build expects.

One entry point matters: :func:`upgrade_to_head`, which
:func:`scufris.db.open_state_database` calls at startup in every process that
opens the database, before any store reads it. It is a no-op on a database
already at head, so the cost of calling it is one query.

Three things here are deliberate:

- **Alembic is configured in code, not from a file.** ``script_location`` comes
  from ``importlib.resources``, so it points inside the INSTALLED package; the
  repo-root ``alembic.ini`` is a maintainer's tool for
  ``alembic revision --autogenerate`` and is not on this path. No
  ``sqlalchemy.url`` is set either - the connection is handed over directly, so
  an ``env.py`` that fell back to opening its own engine would fail loudly
  rather than quietly migrate on SQLite's default pragmas.
- **The revision is asked twice, and only the second question holds a lock.**
  The common startup has nothing to do, so it answers from a raw read that takes
  no write lock at all; taking the exclusive lock merely to discover "already at
  head" would make every start contend with whatever is writing, and
  ``busy_timeout`` turns a wait over five seconds into a hard failure rather
  than a longer wait. When there IS something to do, the revision is re-read
  inside the same ``BEGIN IMMEDIATE`` that applies it, so two processes starting
  together - the dashboard and an orchestrator MCP subprocess both open this
  file - cannot both decide to create the same table.
- **A database with contents is copied off before its schema moves.**
  ``VACUUM INTO`` (20260801-100405 DECISION.md section 4): one statement, one
  consistent file, no coordination with writers. A fresh database has nothing to
  protect and is not copied.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import Connection
from sqlalchemy.exc import OperationalError

from .engine import FILE_MODE, Database

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "scufris.db.migrations"

# How a migration context is configured, in ONE place: `env.py` runs migrations
# under these, and the drift test compares under them. Split in two and a change
# to env.py silently stops the drift test measuring what production does.
#
# render_as_batch because SQLite cannot ALTER most things in place - every ALTER
# has to go through Alembic's copy-and-move. NOTE for whoever writes the first
# ALTER: this connection runs with foreign_keys=ON inside an open transaction,
# where `PRAGMA foreign_keys` is a no-op, so a batch ALTER cannot turn foreign
# keys off for the copy. Batch ALTERs on this path require FK-free tables;
# re-decide this before the first foreign key lands.
MIGRATION_CONTEXT_OPTS: dict[str, bool] = {
    "render_as_batch": True,
    "compare_type": True,
}


def _migrations_dir() -> Path:
    """Where the shipped Alembic environment lives inside this distribution."""
    return Path(str(importlib.resources.files(MIGRATIONS_PACKAGE)))


def _alembic_config() -> Config:
    """An Alembic config pointing at the shipped environment.

    No ``sqlalchemy.url``: the caller puts an open connection on ``attributes``,
    which is the only supported way in.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(_migrations_dir()))
    return cfg


def head_revision() -> str:
    """The revision this build's ``versions/`` ends at."""
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError(f"{MIGRATIONS_PACKAGE} contains no revisions")
    return head


def current_revision(db: Database) -> str | None:
    """The revision the database is at, or ``None`` if it has never been migrated.

    Read on a RAW DBAPI connection, and that is the whole point of the function.
    Every begin on this engine is a ``BEGIN IMMEDIATE`` (see
    ``scufris/db/engine.py``), so asking this through SQLAlchemy would take the
    WRITE lock to answer a read-only question - which is exactly what the startup
    check must not do. In WAL a raw read never blocks and is never blocked.

    The cost of leaving the boundary is the exception TYPE: a damaged database
    raises the driver's ``sqlite3.DatabaseError`` here rather than SQLAlchemy's
    wrapper. Damage still never reads as an empty database, which is the promise;
    ``sqlite3.DatabaseError`` is the type common to both paths.
    """
    raw = db.engine.raw_connection()
    try:
        cursor = raw.cursor()
        try:
            row = cursor.execute("SELECT version_num FROM alembic_version").fetchone()
        except sqlite3.OperationalError as exc:
            # No table means the database has never been migrated. Anything else
            # - a damaged page, an unreadable file - is the caller's to see.
            if "no such table" not in str(exc).lower():
                raise
            return None
        finally:
            cursor.close()
    finally:
        raw.close()
    return None if row is None else str(row[0])


def _current_revision(conn: Connection) -> str | None:
    return MigrationContext.configure(conn).get_current_revision()


def backup_path(database: Path, revision: str) -> Path:
    """Where the pre-migration copy of ``database`` at ``revision`` is written."""
    return database.with_name(f"{database.name}.pre-{revision}.bak")


def backup_database(db: Database, revision: str) -> Path:
    """Copy the database to its pre-migration backup and return where it went.

    ``VACUUM INTO`` rather than a file copy: the copy has to be a whole database,
    and the committed truth in WAL mode is spread across the file and an
    uncheckpointed ``-wal`` sibling that a file copy would miss.

    Run on a raw DBAPI connection because ``VACUUM`` cannot run inside a
    transaction, and every begin on this engine is an immediate one. The
    statement takes only a read snapshot, so it is safe while the caller holds
    the write lock it is about to migrate under.
    """
    target = backup_path(db.path, revision)
    if target.is_symlink():
        raise RuntimeError(f"{target} is a symlink; refusing to write the backup")
    # VACUUM INTO refuses an existing target. A leftover here is a previous
    # migration attempt that failed after the copy; the database has not moved
    # since, so the fresh copy is the same state and replacing it loses nothing.
    target.unlink(missing_ok=True)
    raw = db.engine.raw_connection()
    # SQLite creates the target under the process UMASK, so the mode has to be
    # forced BEFORE the statement, not after it: chmod-ing afterwards leaves a
    # complete copy of the state database - live session ids and all - readable
    # by everyone for as long as the vacuum runs, which on a large database is
    # not a moment. The chmod below stays as a belt-and-braces narrow.
    #
    # umask is PROCESS-global, so this is only safe because migration is a
    # single-threaded startup step that runs before anything else creates files.
    previous_umask = os.umask(0o077)
    try:
        cursor = raw.cursor()
        try:
            cursor.execute("VACUUM INTO ?", (str(target),))
        finally:
            cursor.close()
    finally:
        os.umask(previous_umask)
        raw.close()
    os.chmod(target, FILE_MODE)
    return target


def _known_revision(revision: str) -> bool:
    """Whether this build's ``versions/`` contains ``revision``."""
    try:
        ScriptDirectory.from_config(_alembic_config()).get_revision(revision)
    except CommandError:
        return False
    return True


def upgrade_to_head(db: Database) -> None:
    """Apply every pending revision to ``db``. A no-op if it is already at head.

    Raises ``RuntimeError`` when the database is at a revision this build does
    not have (it was written by a newer Scufris) or when another process is
    holding the write lock for longer than the busy timeout.
    """
    head = head_revision()
    # Asked WITHOUT the write lock first, because the answer is "nothing to do"
    # on every startup after the first. Taking the exclusive lock merely to
    # discover that would make each start contend with whatever is writing, and
    # busy_timeout turns a wait longer than five seconds into a hard failure.
    if current_revision(db) == head:
        return

    try:
        with db.transaction() as conn:
            # Re-read under the lock: between the check above and here another
            # process may have migrated, and this is the read that check-then-act
            # has to make atomic with the upgrade.
            current = _current_revision(conn)
            if current == head:
                return
            if current is not None:
                if not _known_revision(current):
                    raise RuntimeError(
                        f"{db.path} is at revision {current}, which this build of "
                        "Scufris does not have: the database was written by a "
                        "newer version. Install that version, or restore a "
                        "backup taken before it ran."
                    )
                target = backup_database(db, current)
                logger.info("backed up the state database to %s", target)
            logger.info(
                "migrating the state database %s -> %s", current or "empty", head
            )
            cfg = _alembic_config()
            # env.py takes this connection instead of opening its own, so the
            # migration inherits the production pragmas and joins the
            # transaction already begun above.
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
    except OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        raise RuntimeError(
            f"could not migrate {db.path}: another Scufris process is holding "
            "the write lock. Stop it and start again."
        ) from exc
