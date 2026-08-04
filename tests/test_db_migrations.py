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

import ast
import importlib
import importlib.resources
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import DatabaseError, IntegrityError

from scufris.config import Settings
from scufris.db import (
    Database,
    database_path,
    migrate,
    open_database,
    open_state_database,
)
from scufris.db.migrate import (
    MIGRATION_CONTEXT_OPTS,
    MIGRATIONS_PACKAGE,
    SQUASHED_REVISIONS,
    _alembic_config,
    backup_database,
    backup_path,
    current_revision,
    head_revision,
    upgrade_to_head,
)
from scufris.db.models import Base
from scufris_chat import ActorKind


def _startup(state_dir: Path) -> None:
    """What a process does at startup, for a test that does not want the handle."""
    open_state_database(state_dir).close()


def _tables(db: Database) -> set[str]:
    return set(inspect(db.engine).get_table_names())


_FOLLOW_ON_REVISION = '''"""throwaway follow-on

Revision ID: 0000feed0000
Revises: {baseline}

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0000feed0000"
down_revision: str | Sequence[str] | None = "{baseline}"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "throwaway_probe",
        sa.Column("id", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("throwaway_probe")
'''


@pytest.fixture
def behind_head(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> str:
    """A build whose ``versions/`` carries one revision AFTER the shipped one.

    v0.2.0 squashed `versions/` down to a single baseline, so on the shipped tree
    there is no "behind head at a revision this build knows" state to build - and
    the proof that a real upgrade copies the database off first has nowhere to
    stand. Rather than retire that proof until the next revision lands, the
    environment is staged: the shipped `env.py`, `script.py.mako` and `versions/`
    are copied to a tmp directory, one throwaway follow-on revision is written
    into the copy, and `_migrations_dir` is pointed at it. Every entry point in
    `migrate.py` reads that one function, so `head_revision`, `_alembic_config`
    and `upgrade_to_head` all move together.

    Returns the shipped baseline, which is the revision the database is left
    behind at.
    """
    shipped = Path(str(importlib.resources.files(MIGRATIONS_PACKAGE)))
    staged = tmp_path_factory.mktemp("migrations")
    shutil.copytree(
        shipped,
        staged,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    baseline = head_revision()
    (staged / "versions" / "0000feed0000_throwaway_follow_on.py").write_text(
        _FOLLOW_ON_REVISION.format(baseline=baseline)
    )
    monkeypatch.setattr(migrate, "_migrations_dir", lambda: staged)
    assert head_revision() == "0000feed0000"
    return baseline


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


def test_the_backup_is_taken_on_the_real_migration_path(
    tmp_path: Path, behind_head: str
) -> None:
    """A database BEHIND head is copied off before the revision that moves it.

    The database is brought to the revision before head, given a row, and then
    upgraded the way startup does it. The copy has to be the state as it was
    BEFORE - the old revision, and none of the tables the new one adds - or it is
    not a rollback target.

    The `behind_head` fixture supplies the second revision the squashed tree no
    longer ships. Nothing about the property under test depends on WHAT that
    revision does, only that head is one step past where the database sits.
    """
    previous = behind_head

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
        # The table the HEAD revision adds, so this discriminates a real
        # pre-migration copy from a copy taken afterwards.
        assert ("throwaway_probe",) not in tables
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


@pytest.mark.parametrize("revision", sorted(SQUASHED_REVISIONS))
def test_a_pre_v020_database_is_refused_with_delete_instructions(
    tmp_path: Path, revision: str
) -> None:
    """A v0.1.x database is BEHIND this build, not ahead of it.

    Its five revisions were squashed into the v0.2.0 baseline, so the generic
    unknown-revision refusal would tell the operator the opposite of the truth -
    "install the newer version" for a database an older one wrote. v0.2.0 carries
    no data forward, so the only true instruction is to delete it. No backup is
    written for a database nothing is going to migrate.
    """
    db = open_database(tmp_path)
    try:
        upgrade_to_head(db)
        with db.transaction() as conn:
            conn.execute(
                text("UPDATE alembic_version SET version_num=:rev"), {"rev": revision}
            )

        with pytest.raises(RuntimeError, match="v0.1.x") as excinfo:
            upgrade_to_head(db)
    finally:
        db.close()

    assert "Delete the database" in str(excinfo.value)
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

    This is now every app-owned store: the projects and agent-state halves, the
    auth, schedule, digest and host-action tables 20260801-100413 added, the
    config-change table 20260803-002141 closed the boundary with, and the
    `conversation` and `event` tables 20260804-115256 opened `packages/chat`
    with. The `delivery` and `activity` tables the epic anticipates are NOT here
    - a table for one appearing would mean a revision was written against a
    model nothing reads yet.

    The chat pair is listed here AND asserted in detail by
    `test_migration_creates_the_chat_tables`, which is not a duplicate: this
    test says nothing else arrived, and that one says what arrived is right.
    """
    upgrade_to_head(fresh)

    assert _tables(fresh) == {
        "alembic_version",
        "projects",
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
        "config_change",
        "conversation",
        "event",
    }


def test_migration_creates_the_chat_tables(fresh: Database) -> None:
    """The SHIPPED migration builds `packages/chat`'s tables, constraints and all.

    `packages/chat/tests` creates its tables from `Base.metadata`, which proves
    the store agrees with the models and nothing about what an operator's
    database actually gets. This is that half: a fresh file taken to head by the
    real runner, then asked whether the two invariants that live in the SCHEMA
    survived the revision - the uniqueness of `(conversation_id, event_seq)` and
    the CHECK that pins `actor_kind` to four values.

    Both are asserted by INSERTING against them rather than by reading the DDL
    back: a constraint SQLite parsed but does not enforce would still appear in
    `sqlite_master`.
    """
    upgrade_to_head(fresh)

    assert {"conversation", "event"} <= _tables(fresh)

    def insert_event(**overrides: object) -> None:
        values: dict[str, object] = {
            "id": "e1",
            "conversation_id": "c1",
            "event_seq": 1,
            "actor_kind": "operator",
            "actor_agent_id": None,
            "kind": "message",
            "body": "hello",
            "correlation_id": None,
            "causation_id": None,
            "created_at": 0.0,
        }
        values.update(overrides)
        with fresh.transaction() as conn:
            conn.execute(
                text(
                    "INSERT INTO event (id, conversation_id, event_seq, actor_kind, "
                    "actor_agent_id, kind, body, correlation_id, causation_id, "
                    "created_at) VALUES (:id, :conversation_id, :event_seq, "
                    ":actor_kind, :actor_agent_id, :kind, :body, :correlation_id, "
                    ":causation_id, :created_at)"
                ),
                values,
            )

    with fresh.transaction() as conn:
        conn.execute(
            text("INSERT INTO conversation (id, created_at) VALUES ('c1', 0.0)")
        )
    insert_event()

    with pytest.raises(IntegrityError):
        insert_event(id="e2")
    with pytest.raises(IntegrityError):
        insert_event(id="e3", event_seq=2, actor_kind="wizard")

    # The actor rule is BOTH halves: only an `agent` names an agent, and an
    # `agent` always does. A row that satisfies the kind check alone would make
    # `read_transcript` raise for the whole conversation, not just that row.
    with pytest.raises(IntegrityError):
        insert_event(id="e5", event_seq=3, actor_agent_id="smuggled")
    with pytest.raises(IntegrityError):
        insert_event(id="e6", event_seq=4, actor_kind="agent", actor_agent_id=None)
    # The rule is truthiness, not nullability: `Actor` refuses an empty id as
    # readily as a missing one, and a repair INSERT with an uninterpolated
    # variable produces `''` more readily than it produces a name.
    with pytest.raises(IntegrityError):
        insert_event(id="e8", event_seq=6, actor_kind="agent", actor_agent_id="")
    with pytest.raises(IntegrityError):
        insert_event(id="e9", event_seq=7, actor_agent_id="")
    insert_event(id="e7", event_seq=5, actor_kind="agent", actor_agent_id="builder")

    # The same seq under a DIFFERENT conversation is fine - the constraint is
    # per-conversation, which is the whole point of the pair.
    insert_event(id="e4", conversation_id="c2")


def test_migrated_actor_check_lists_exactly_the_declared_kinds(
    fresh: Database,
) -> None:
    """The SHIPPED check text names the same four kinds `ActorKind` does.

    `models.py` renders its constraint from the enum, so the two cannot drift on
    the model side, and the package suite builds its tables from `Base.metadata`.
    Neither reaches the revision: Alembic's `compare_metadata` does not diff
    CHECK constraints, so a fifth kind would be green everywhere - including
    `test_schema_has_no_pending_autogenerate_diff` - and raise `IntegrityError`
    only on a migrated operator database. This is the one assertion that reads
    what an operator's file actually enforces.
    """
    upgrade_to_head(fresh)

    with fresh.transaction() as conn:
        ddl = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'event'"
            )
        ).scalar_one()

    quoted = re.search(r"actor_kind IN \(([^)]*)\)", ddl)
    assert quoted is not None, ddl
    listed = tuple(part.strip().strip("'") for part in quoted.group(1).split(","))
    assert listed == tuple(kind.value for kind in ActorKind)


def test_a_damaged_database_raises_at_startup_rather_than_reading_as_empty(
    tmp_path: Path,
) -> None:
    """Startup RAISES on a corrupt database rather than reading it as empty.

    WHERE it lands is not fixed, and must not be asserted as if it were: whether
    the pragma dial `open_database` does (SQLAlchemy, so the wrapper) or the
    revision read that follows (a raw connection, so the driver's own error) is
    the one to touch a damaged page depends on how many pages the schema
    occupies. It moved from the second to the first when `config_change` was
    added. Both types are accepted; what must never happen is the other outcome
    entirely - a corrupt database answering "never migrated" and being silently
    migrated again.
    """
    _startup(tmp_path)

    path = database_path(tmp_path)
    raw = bytearray(path.read_bytes())
    assert len(raw) > 4096, "the fixture needs pages past the header to corrupt"
    raw[4096:] = b"\x00" * (len(raw) - 4096)
    path.write_bytes(bytes(raw))

    with pytest.raises((sqlite3.DatabaseError, DatabaseError), match="malformed"):
        _startup(tmp_path)


def _package_model_modules() -> dict[str, Path]:
    """Import name -> file, for every `models.py` a workspace member ships."""
    packages = Path(__file__).resolve().parent.parent / "packages"
    found: dict[str, Path] = {}
    for path in sorted(packages.glob("*/src/*/models.py")):
        found[f"{path.parent.name}.models"] = path
    return found


def _tablenames(path: Path) -> set[str]:
    """The `__tablename__` literals a module assigns, read without importing it."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__tablename__"
            for target in targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            names.add(node.value.value)
    return names


def _env_imports() -> list[str]:
    """The absolute module names `scufris/db/migrations/env.py` imports.

    Read with `ast` rather than by importing the module: `env.py` binds
    `alembic.context`, which only exists inside an alembic run.
    """
    env = Path(__file__).resolve().parent.parent / "scufris" / "db" / "migrations"
    names: list[str] = []
    for node in ast.walk(ast.parse((env / "env.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


#: Imports `env.py`'s module list into a BARE interpreter and prints the tables
#: that ends up registering, one per line. Run as `python -c`, so it carries no
#: import of its own beyond the list it is given.
_REGISTERED_TABLES = """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
print("\\n".join(sorted(importlib.import_module("scufris.db.models").Base.metadata.tables)))
"""


def _tables_env_registers() -> set[str]:
    """`Base.metadata`'s tables after importing exactly what `env.py` imports.

    In a SUBPROCESS, which is the whole point. Importing the list in-process
    measures nothing under a full run: `scufris_hostctl` is already in
    `sys.modules` by then - the app, the example test and the package suite all
    import it - so every table is registered whatever `env.py` says, and the
    check passes with the import deleted. A fresh interpreter sees only what
    `env.py` names.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _REGISTERED_TABLES, *_env_imports()],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return set(completed.stdout.split())


def test_every_package_model_is_registered() -> None:
    """A package's tables reach `Base.metadata` through what `env.py` imports.

    The narrow claim: a workspace member that declares tables and is NOT imported
    by the migration environment has tables that are never created. Every read
    and write against them then fails at runtime - a broken feature, found by an
    operator instead of by this run. It is not data-loss protection; v0.2.0 has
    no operator data to lose.

    Both sides are derived, so neither can quietly agree with the other. The
    expected tablenames are read off each package's SOURCE with `ast`. The
    metadata comes from a fresh interpreter given exactly the module list
    `env.py` itself names - not a list repeated here, which would pass while
    `env.py` was empty, and not this interpreter, which has already imported
    every package for other reasons.
    """
    modules = _package_model_modules()
    assert modules, "no packages/*/src/*/models.py found; the check would be vacuous"

    registered = _tables_env_registers()
    missing = {
        name: sorted(_tablenames(path) - registered)
        for name, path in modules.items()
        if _tablenames(path) - registered
    }
    assert missing == {}, (
        f"tables absent from the migration metadata: {missing}. Add the module to "
        "scufris/db/migrations/env.py, or its tables are never created."
    )
