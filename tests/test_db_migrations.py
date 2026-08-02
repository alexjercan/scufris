"""The schema and the thing that applies it, proven before any store needs it.

Two of these discriminate rather than restate. The autogenerate proof is the one
that catches a hand-edited revision drifting from `models.py` - the failure mode
a migration framework exists to prevent, and the one a "does the table exist"
assertion sails straight past. The pragma proof catches an `env.py` that opens
its OWN engine from a URL: that migrates the right file with SQLite's DEFAULTS -
rollback journal, no busy timeout, foreign keys off - and every table it creates
still looks correct afterwards.
"""

from __future__ import annotations

import importlib.resources
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import event, inspect, text

from scufris.config import Settings
from scufris.db import (
    Database,
    database_path,
    open_database,
    open_state_database,
)
from scufris.db.migrate import (
    MIGRATION_CONTEXT_OPTS,
    _alembic_config,
    backup_database,
    backup_path,
    current_revision,
    head_revision,
    upgrade_to_head,
)
from scufris.db.models import Base


def _startup(state_dir: Path) -> None:
    """What a process does at startup, for a test that does not want the handle."""
    open_state_database(state_dir).close()


def _tables(db: Database) -> set[str]:
    return set(inspect(db.engine).get_table_names())


def _previous_revision() -> str:
    """The revision before head, so a test can build a database that is behind."""
    script = ScriptDirectory.from_config(_alembic_config())
    down = script.get_revision(head_revision()).down_revision
    assert isinstance(down, str), "head has no single parent revision"
    return down


def _upgrade_to(db: Database, revision: str) -> None:
    """Migrate to a NAMED revision, the way `upgrade_to_head` reaches head."""
    with db.transaction() as conn:
        cfg = _alembic_config()
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)


@pytest.fixture
def fresh(tmp_path: Path) -> Iterator[Database]:
    """A database that has NEVER been migrated.

    The shared `database` fixture is already at head, which is what every store
    test wants and what every proof in this module has to do without: a runner
    cannot be shown to reach head from a database that starts there.
    """
    db = open_database(tmp_path)
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Reaching head
# --------------------------------------------------------------------------


def test_migrations_reach_head_and_are_idempotent(fresh: Database) -> None:
    """A fresh state dir reaches head, and a second startup changes nothing."""
    upgrade_to_head(fresh)

    assert current_revision(fresh) == head_revision()
    assert "projects" in _tables(fresh)

    before = _tables(fresh)
    upgrade_to_head(fresh)

    assert current_revision(fresh) == head_revision()
    assert _tables(fresh) == before


def test_schema_has_no_pending_autogenerate_diff(fresh: Database) -> None:
    """The declarative models and the MIGRATED database agree.

    This is the proof that a revision written by hand (or an edited one) cannot
    silently drift from `models.py`: autogenerate is asked what it would still
    have to do, and the answer has to be nothing.
    """
    upgrade_to_head(fresh)

    with fresh.engine.connect() as conn:
        # The options env.py itself runs under, not a hand-typed copy: comparing
        # under different options would measure something production never does.
        context = MigrationContext.configure(conn, opts=dict(MIGRATION_CONTEXT_OPTS))
        diff = compare_metadata(context, Base.metadata)

    assert diff == []


def test_projects_table_matches_the_project_record(fresh: Database) -> None:
    """The one table this task creates carries exactly the `Project` fields."""
    upgrade_to_head(fresh)

    inspector = inspect(fresh.engine)
    columns = {c["name"]: c for c in inspector.get_columns("projects")}

    assert set(columns) == {"id", "cwd", "name", "language", "description"}
    assert not any(c["nullable"] for c in columns.values())
    assert inspector.get_pk_constraint("projects")["constrained_columns"] == ["id"]


# --------------------------------------------------------------------------
# The connection the migration runs on
# --------------------------------------------------------------------------


def test_migration_connection_uses_production_pragmas(fresh: Database) -> None:
    """The migration runs on the app's OWN engine, not one `env.py` dialed itself.

    The DDL ITSELF is what has to be seen on this engine, not merely some traffic
    on it: the runner reads the current revision here before it migrates, so a
    test that accepted any captured statement would still pass against an
    `env.py` that then went off and dialled its own engine - measured, it did.
    The pragmas are read back off the captured DBAPI connection, because reading
    them through SQLAlchemy would begin a transaction on an engine whose every
    begin is immediate.
    """
    creating: list[sqlite3.Connection] = []

    @event.listens_for(fresh.engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        if "CREATE TABLE projects" in statement:
            creating.append(conn.connection.dbapi_connection)

    try:
        upgrade_to_head(fresh)
    finally:
        event.remove(fresh.engine, "before_cursor_execute", _capture)

    assert creating, "the schema was not created on the app's own engine"

    cursor = creating[0].cursor()
    try:
        pragmas = {
            name: cursor.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("journal_mode", "synchronous", "busy_timeout", "foreign_keys")
        }
    finally:
        cursor.close()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 2,
        "busy_timeout": 5000,
        "foreign_keys": 1,
    }


# --------------------------------------------------------------------------
# The backup taken before a schema migration
# --------------------------------------------------------------------------


def test_a_fresh_database_is_not_backed_up(tmp_path: Path) -> None:
    """There is nothing to protect on the first startup, so nothing is copied.

    The revision assertion is the delivery guard: without it this passes just as
    happily if the startup migration became a no-op, which is the one way "no
    backup was written" stops meaning anything.
    """
    _startup(tmp_path)

    db = open_database(tmp_path)
    try:
        assert current_revision(db) == head_revision()
    finally:
        db.close()
    assert list(tmp_path.glob("*.bak")) == []


def test_a_database_at_head_is_neither_migrated_nor_backed_up(tmp_path: Path) -> None:
    """The second startup does nothing at all - no revision moved, no copy."""
    _startup(tmp_path)
    db = open_database(tmp_path)
    try:
        before = current_revision(db)
    finally:
        db.close()
    assert before == head_revision()

    _startup(tmp_path)

    db = open_database(tmp_path)
    try:
        assert current_revision(db) == before
    finally:
        db.close()
    assert list(tmp_path.glob("*.bak")) == []


def test_the_backup_is_a_whole_readable_database(tmp_path: Path) -> None:
    """The pre-migration copy carries the committed rows and the file's own mode.

    Exercised directly rather than through `upgrade_to_head`, which is where the
    property itself lives: that the copy is a complete database rather than a
    file-level snapshot of a WAL that was never checkpointed, and that it is not
    left world-readable. The WIRING - that a real upgrade takes one first - is
    `test_the_backup_is_taken_on_the_real_migration_path`.
    """
    db = open_database(tmp_path)
    try:
        upgrade_to_head(db)
        with db.transaction() as conn:
            conn.execute(
                text(
                    "INSERT INTO projects (id, cwd, name, language, description) "
                    "VALUES ('p', '/tmp', 'P', '', '')"
                )
            )
        revision = current_revision(db)
        assert revision is not None

        target = backup_database(db, revision)
    finally:
        db.close()

    assert target == backup_path(database_path(tmp_path), revision)
    assert target.stat().st_mode & 0o777 == 0o600
    copy = sqlite3.connect(target)
    try:
        assert copy.execute("SELECT id FROM projects").fetchall() == [("p",)]
        assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        copy.close()


def test_the_backup_is_taken_on_the_real_migration_path(tmp_path: Path) -> None:
    """A database BEHIND head is copied off before the revision that moves it.

    The database is brought to the revision before head, given a row, and then
    upgraded the way startup does it. The copy has to be the state as it was
    BEFORE - the old revision, and none of the tables the new one adds - or it is
    not a rollback target.
    """
    previous = _previous_revision()

    db = open_database(tmp_path)
    try:
        _upgrade_to(db, previous)
        with db.transaction() as conn:
            conn.execute(
                text(
                    "INSERT INTO projects (id, cwd, name, language, description) "
                    "VALUES ('p', '/tmp', 'P', '', '')"
                )
            )

        upgrade_to_head(db)

        assert current_revision(db) == head_revision()
    finally:
        db.close()

    copy = sqlite3.connect(backup_path(database_path(tmp_path), previous))
    try:
        assert copy.execute("SELECT id FROM projects").fetchall() == [("p",)]
        assert copy.execute("SELECT version_num FROM alembic_version").fetchone() == (
            previous,
        )
        tables = copy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        # A table the HEAD revision adds, so this discriminates a real
        # pre-migration copy from a copy taken afterwards.
        assert ("host_action",) not in tables
    finally:
        copy.close()


def test_the_backup_is_never_world_readable_even_briefly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy is 0600 as SQLite CREATES it, not 0600 once we get round to it.

    `VACUUM INTO` creates the target under the process umask, so a chmod
    AFTERWARDS leaves a complete copy of the state database - live session ids
    and all - readable by everyone for as long as the vacuum runs (measured at
    0644 before the fix, under `umask 022`).

    The chmod is disabled for the duration rather than the window timed: the
    mode the file is created with is the property, and a timing probe on a
    database this small would be racing the very window it is trying to observe.
    """
    monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: None)
    previous = os.umask(0o022)
    try:
        db = open_database(tmp_path)
        try:
            upgrade_to_head(db)
            revision = current_revision(db)
            assert revision is not None
            target = backup_database(db, revision)
        finally:
            db.close()
    finally:
        os.umask(previous)

    assert target.stat().st_mode & 0o777 == 0o600


def test_a_database_at_head_does_not_take_the_write_lock(tmp_path: Path) -> None:
    """The common startup asks the question without contending for the answer.

    Every start after the first has nothing to do, and `busy_timeout` turns a
    wait longer than five seconds into a hard failure - so taking the exclusive
    lock merely to discover "already at head" would make each start fail against
    whatever happens to be writing. Proven by holding the write lock throughout.

    The lock is held by a raw `sqlite3` connection, not a second `Database`: the
    real case is two PROCESSES, and two `Database` objects in one process share
    the `transaction()` nesting guard, so the sabotage would trip that guard
    instead of the lock and the test would pass for the wrong reason. A raw
    connection reproduces cross-process contention exactly.
    """
    _startup(tmp_path)

    holder = sqlite3.connect(database_path(tmp_path), isolation_level=None)
    holder.execute("PRAGMA busy_timeout=5000")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO projects (id, cwd, name, language, description) "
        "VALUES ('p', '/tmp', 'P', '', '')"
    )
    try:
        db = open_database(tmp_path)
        try:
            started = time.monotonic()
            upgrade_to_head(db)
            elapsed = time.monotonic() - started
        finally:
            db.close()
    finally:
        holder.rollback()
        holder.close()

    assert elapsed < 1.0, f"the at-head path contended for {elapsed:.1f}s"


def test_a_database_from_a_newer_scufris_is_refused_without_a_backup(
    tmp_path: Path,
) -> None:
    """An unknown revision is a newer Scufris, not something to migrate forward.

    Before the check it was treated as merely "behind head": the backup was
    written first, and only then did Alembic fail with `Can't locate revision`,
    leaving a stray `.bak` named after a revision this build knows nothing about.
    """
    db = open_database(tmp_path)
    try:
        upgrade_to_head(db)
        with db.transaction() as conn:
            conn.execute(text("UPDATE alembic_version SET version_num='deadbeefcafe'"))

        with pytest.raises(RuntimeError, match="written by a newer version"):
            upgrade_to_head(db)
    finally:
        db.close()

    assert list(tmp_path.glob("*.bak")) == []


# --------------------------------------------------------------------------
# Where it runs from
# --------------------------------------------------------------------------


def test_migration_scripts_ship_inside_the_package() -> None:
    """The environment is importable package DATA, not a repo-root directory.

    The wheel is built with `only-include = ["scufris"]`, so a root `alembic/`
    would simply not reach an operator.
    """
    import scufris

    root = Path(str(importlib.resources.files("scufris.db.migrations")))

    # Under the `scufris` package itself, wherever that package happens to be.
    # This clause cannot fail on its own - `importlib.resources.files` resolves
    # there by construction - and is kept only to say what the layout is. The
    # proof that the files actually SHIP is DoD command 4, which reads the built
    # store path; nothing runnable inside the repo can distinguish that.
    assert root.is_relative_to(Path(scufris.__file__).parent)
    assert (root / "env.py").is_file()
    assert (root / "script.py.mako").is_file()
    assert any(
        entry.name.endswith(".py") and entry.name != "__init__.py"
        for entry in (root / "versions").iterdir()
    )


def test_app_startup_upgrades_state_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """App construction migrates the database BEFORE the first store is built.

    The ordering is the point, so it is what is measured: the first store's
    constructor is wrapped and asked what revision the database was at when it
    ran. Asserting only on the finished app would pass just as happily with the
    migration wired in after every store.
    """
    from scufris import app as app_module

    at_construction: list[str | None] = []
    real = app_module.ProjectStore

    def spy(settings: Settings, db: Database) -> object:
        # A SECOND handle on the same file, so what is measured is the schema on
        # DISK when the store is built, not what the app's own handle knows.
        observer = open_database(Path(settings.state_dir))
        try:
            at_construction.append(current_revision(observer))
        finally:
            observer.close()
        return real(settings, db)

    monkeypatch.setattr(app_module, "ProjectStore", spy)

    settings = Settings(state_dir=tmp_path)
    app_module.create_app(settings=settings)

    assert at_construction == [head_revision()]

    db = open_database(tmp_path)
    try:
        assert "projects" in _tables(db)
    finally:
        db.close()


def test_migrating_a_missing_state_dir_creates_it(tmp_path: Path) -> None:
    """A first run has no state directory at all."""
    state_dir = tmp_path / "nested" / "state"

    _startup(state_dir)

    db = open_database(state_dir)
    try:
        assert current_revision(db) == head_revision()
    finally:
        db.close()


def test_declared_tables_are_the_only_ones(fresh: Database) -> None:
    """The whole schema, listed once, so an unreviewed table cannot arrive quietly.

    This is now every app-owned store: the projects and agent-state halves, and
    the auth, schedule, digest and host-action tables 20260801-100413 added. The
    conversation and activity tables the epic anticipates are NOT here - a table
    for one appearing would mean a revision was written against a model nothing
    reads yet.

    `legacy_import` is bookkeeping for the one-way JSON import, not a store.
    """
    upgrade_to_head(fresh)

    assert _tables(fresh) == {
        "alembic_version",
        "projects",
        "legacy_import",
        "agents",
        "agent_session",
        "agent_session_history",
        "agent_outcome",
        "settings_override",
        "reasoning_turn",
        "auth_session",
        "schedule",
        "digest",
        "host_action",
    }


def test_a_damaged_database_raises_at_startup_rather_than_reading_as_empty(
    tmp_path: Path,
) -> None:
    """The startup revision read is the FIRST read, so it is where damage lands.

    It reads on a raw connection, so the error arrives as the driver's
    `sqlite3.DatabaseError` rather than SQLAlchemy's wrapper (see
    `current_revision`). What must never happen is the other outcome: a corrupt
    database answering "never migrated" and being silently migrated again.
    """
    _startup(tmp_path)

    path = database_path(tmp_path)
    raw = bytearray(path.read_bytes())
    assert len(raw) > 4096, "the fixture needs pages past the header to corrupt"
    raw[4096:] = b"\x00" * (len(raw) - 4096)
    path.write_bytes(bytes(raw))

    with pytest.raises(sqlite3.DatabaseError, match="malformed"):
        _startup(tmp_path)
