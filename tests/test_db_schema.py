"""What a migrated database actually CONTAINS, as opposed to how it got there.

`test_db_migrations.py` is the runner: reaching head, the connection the DDL
runs on, the backup taken first, where the scripts ship from. This module is the
other half - the shape the revisions leave behind, read off a database the real
runner took to head.

Two of these discriminate rather than restate. The autogenerate proof catches a
hand-edited revision drifting from `models.py`, which is the failure mode a
migration framework exists to prevent. The constraint proofs INSERT against
their constraints rather than reading the DDL back, because a constraint SQLite
parsed but does not enforce would still appear in `sqlite_master`.
"""

from __future__ import annotations

import re

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from scufris.db import Database
from scufris.db.migrate import MIGRATION_CONTEXT_OPTS, upgrade_to_head
from scufris.db.models import Base
from scufris_chat import ActorKind, DeliveryState


def _tables(db: Database) -> set[str]:
    return set(inspect(db.engine).get_table_names())


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
    """The one table the first revision creates carries exactly the `Project` fields."""
    upgrade_to_head(fresh)

    inspector = inspect(fresh.engine)
    columns = {c["name"]: c for c in inspector.get_columns("projects")}

    assert set(columns) == {"id", "cwd", "name", "language", "description"}
    assert not any(c["nullable"] for c in columns.values())
    assert inspector.get_pk_constraint("projects")["constrained_columns"] == ["id"]


def test_declared_tables_are_the_only_ones(fresh: Database) -> None:
    """The whole schema, listed once, so an unreviewed table cannot arrive quietly.

    This is now every app-owned store: the projects and agent-state halves, the
    auth, schedule, digest and host-action tables 20260801-100413 added, the
    config-change table 20260803-002141 closed the boundary with, the
    `conversation` and `event` tables 20260804-115256 opened `packages/chat`
    with, and the `delivery` table 20260804-115319 added to it. The `activity`
    table the epic anticipates is NOT here - it appearing would mean a revision
    was written against a model nothing reads yet.

    The chat tables are listed here AND asserted in detail by
    `test_migration_creates_the_chat_tables` and
    `test_migration_creates_the_delivery_table`, which is not a duplicate: this
    test says nothing else arrived, and those say what arrived is right.
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
        "delivery",
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


def test_migration_creates_the_delivery_table(fresh: Database) -> None:
    """The SHIPPED migration builds `delivery`, composite key and all three CHECKs.

    Same split as `test_migration_creates_the_chat_tables`: the package suite
    builds its tables from `Base.metadata`, which proves the store agrees with
    the models and nothing about what an operator's database gets.

    Asserted by INSERTING against the constraints rather than by reading the DDL
    back - a constraint SQLite parsed but does not enforce would still appear in
    `sqlite_master`. The composite key is what makes a retry after a crash
    collide rather than post a second card, so a key that came through as
    `channel` alone, or as no key at all, has to be visible here.
    """
    upgrade_to_head(fresh)

    assert "delivery" in _tables(fresh)

    def insert_delivery(**overrides: object) -> None:
        values: dict[str, object] = {
            "channel": "telegram",
            "conversation_id": "c1",
            "event_seq": 1,
            "state": "claimed",
            "claimed_at": 0.0,
            "confirmed_at": None,
        }
        values.update(overrides)
        with fresh.transaction() as conn:
            conn.execute(
                text(
                    "INSERT INTO delivery (channel, conversation_id, event_seq, "
                    "state, claimed_at, confirmed_at) VALUES (:channel, "
                    ":conversation_id, :event_seq, :state, :claimed_at, "
                    ":confirmed_at)"
                ),
                values,
            )

    insert_delivery()

    with pytest.raises(IntegrityError):
        insert_delivery()
    with pytest.raises(IntegrityError):
        insert_delivery(event_seq=2, state="sent")

    # The state rule is BOTH halves: a `confirmed` row always carries when, and
    # a `claimed` one never does. Half the rule would let a delivery claim it
    # completed at no time.
    with pytest.raises(IntegrityError):
        insert_delivery(event_seq=3, state="confirmed", confirmed_at=None)
    with pytest.raises(IntegrityError):
        insert_delivery(event_seq=4, state="claimed", confirmed_at=1.0)
    insert_delivery(event_seq=5, state="confirmed", confirmed_at=1.0)

    # Every part of the key discriminates: the same event on a different
    # channel, and the same channel on a different conversation, are both other
    # deliveries rather than this one again.
    insert_delivery(channel="web")
    insert_delivery(conversation_id="c2")

    # An empty channel is a DISTINCT key, not a malformed one, so without the
    # check the row lands under a channel nothing polls while the real channel
    # still sees the event as pending.
    with pytest.raises(IntegrityError):
        insert_delivery(channel="")


def test_migrated_delivery_check_lists_exactly_the_declared_states(
    fresh: Database,
) -> None:
    """The SHIPPED check text names the same two states `DeliveryState` does.

    The `actor_kind` sibling above records why this is its own test, and it
    applies here unchanged: `models.py` renders the constraint from the enum, so
    the two cannot drift on the model side, and the package suite builds its
    tables from `Base.metadata`. Neither reaches the revision - Alembic's
    `compare_metadata` does not diff CHECK constraints - so a third
    `DeliveryState` would be green in the models, in the package tests AND in
    `test_schema_has_no_pending_autogenerate_diff`, and raise `IntegrityError`
    only on a migrated operator database.
    """
    upgrade_to_head(fresh)

    with fresh.transaction() as conn:
        ddl = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'delivery'"
            )
        ).scalar_one()

    quoted = re.search(r"state IN \(([^)]*)\)", ddl)
    assert quoted is not None, ddl
    listed = tuple(part.strip().strip("'") for part in quoted.group(1).split(","))
    assert listed == tuple(state.value for state in DeliveryState)


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
