"""The four claims the conversation tables are for.

Each one is a property of the DATA, not of the code that happens to write it:
the sequence is gap-free because it is read and written inside one immediate
transaction, the actor is typed because the column is CHECK-constrained, and
causation resolves because the event carries the id of what caused it. That is
why three of the four reach for a real file and real threads rather than a
double - a double would prove the store agrees with itself.

`tasks/20260804-115256/DECISION.md` is what these assert; sections 1, 2 and 4.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from scufris_chat import (
    Actor,
    ActorKind,
    append_event,
    causing_event,
    create_conversation,
    read_transcript,
)
from scufris_core import Base, Database, open_database

#: The tables this package owns, created straight from the shared metadata.
OWNED_TABLES = ("conversation", "event")

OPERATOR = Actor(ActorKind.OPERATOR)
BUILDER = Actor(ActorKind.AGENT, "builder")


class Boom(RuntimeError):
    """Raised inside a transaction to make the rollback real."""


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """A file-backed state database holding this package's tables.

    Owned here rather than borrowed, and created from the metadata rather than
    by running Alembic, for the reasons `packages/hostctl/tests` records: the
    migration environment ships with the ROOT distribution, and a package suite
    that ran it would depend on the app to test a store boundary.
    `tests/test_db_migrations.py::test_migration_creates_the_chat_tables` is
    what proves the shipped migration agrees with this metadata.

    File-backed rather than `:memory:` because the concurrency test needs two
    connections to contend on one write lock, which `:memory:` cannot express.
    """
    db = open_database(tmp_path)
    try:
        Base.metadata.create_all(
            db.engine,
            tables=[Base.metadata.tables[name] for name in OWNED_TABLES],
        )
        yield db
    finally:
        db.close()


def test_event_seq_is_monotonic_under_concurrent_writers(database: Database) -> None:
    """Eight threads, five events each, and forty consecutive numbers.

    REAL threads on a REAL file: the claim is that two units of work cannot
    claim one number, and the only thing that makes it true is that the begin is
    immediate, so the read-modify-write of `MAX(event_seq) + 1` cannot
    interleave. A single-threaded test would pass against a store that assigned
    the number before opening the transaction, which is the defect this guards.

    Synchronous, with no event loop anywhere: `Database.transaction()` refuses a
    thread that has one.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)

    writers, per_writer = 8, 5
    # A barrier so every thread is inside `write` before any of them takes the
    # write lock; without it the threads start far enough apart to serialise on
    # their own and the contention is never reached.
    start = threading.Barrier(writers)
    failures: list[BaseException] = []

    def write(writer: int) -> None:
        start.wait()
        for index in range(per_writer):
            try:
                with database.transaction() as conn:
                    append_event(
                        conn,
                        conversation.id,
                        actor=OPERATOR,
                        kind="message",
                        body=f"writer {writer} event {index}",
                    )
            except BaseException as exc:  # noqa: BLE001 - reported, not handled
                failures.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert [t for t in threads if t.is_alive()] == []
    assert failures == []

    with database.transaction() as conn:
        transcript = read_transcript(conn, conversation.id)
    assert [event.event_seq for event in transcript] == list(
        range(1, writers * per_writer + 1)
    )


def test_rolled_back_event_consumes_no_seq(database: Database) -> None:
    """A unit of work that unwinds leaves no hole behind it.

    This is the same property from the other side: the number is read INSIDE the
    transaction that inserts the event, so an aborted write takes its candidate
    number down with it. A counter kept anywhere else - a column on the
    conversation bumped first, an in-process integer - would leave a gap here.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        first = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="one"
        )
    assert first.event_seq == 1

    with pytest.raises(Boom):
        with database.transaction() as conn:
            append_event(
                conn, conversation.id, actor=OPERATOR, kind="message", body="lost"
            )
            raise Boom("something went wrong after the append")

    with database.transaction() as conn:
        third = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="two"
        )
    assert third.event_seq == 2

    with database.transaction() as conn:
        transcript = read_transcript(conn, conversation.id)
    assert [event.body for event in transcript] == ["one", "two"]
    assert [event.event_seq for event in transcript] == [1, 2]


def test_actor_must_be_a_known_kind(database: Database) -> None:
    """An actor is a typed value, and the column refuses what the parse would.

    Two gates, because they cover different attackers. `Actor.parse` is the one
    boundary a string crosses, so an unknown kind, an `agent` without an id and
    a non-agent WITH one all raise there. The CHECK constraint is what a
    hand-written INSERT - a migration, a repair session, a future store that
    forgot - meets instead.
    """
    with pytest.raises(ValueError, match="wizard"):
        Actor.parse("wizard")
    with pytest.raises(ValueError, match="id"):
        Actor.parse("agent")
    with pytest.raises(ValueError, match="id"):
        Actor.parse("operator:alex")
    # The separator with nothing after it is a malformed wire form, not a bare
    # kind: `operator:` says an id follows and then names none.
    with pytest.raises(ValueError, match="id"):
        Actor.parse("operator:")
    with pytest.raises(ValueError, match="id"):
        Actor.parse("agent:")

    assert Actor.parse("agent:builder") == BUILDER
    assert Actor.parse("operator") == OPERATOR

    with database.transaction() as conn:
        conversation = create_conversation(conn)
        append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="build it"
        )
        append_event(conn, conversation.id, actor=BUILDER, kind="report", body="built")

    with database.transaction() as conn:
        transcript = read_transcript(conn, conversation.id)
    assert [event.actor for event in transcript] == [OPERATOR, BUILDER]

    def forge(actor_kind: str, actor_agent_id: str | None, seq: int) -> None:
        with database.transaction() as conn:
            conn.execute(
                text(
                    "INSERT INTO event (id, conversation_id, event_seq, actor_kind, "
                    "actor_agent_id, kind, body, correlation_id, causation_id, "
                    "created_at) VALUES ('forged', :conversation, :seq, :kind, "
                    ":agent_id, 'message', 'let me in', NULL, NULL, 0.0)"
                ),
                {
                    "conversation": conversation.id,
                    "seq": seq,
                    "kind": actor_kind,
                    "agent_id": actor_agent_id,
                },
            )

    with pytest.raises(IntegrityError):
        forge("wizard", None, 99)

    # BOTH halves of the actor rule are the schema's, not just the kind. One row
    # whose kind and id disagree would make `read_transcript` raise for the
    # WHOLE conversation, since every row is rebuilt into an `Actor`.
    with pytest.raises(IntegrityError):
        forge("operator", "smuggled", 98)
    with pytest.raises(IntegrityError):
        forge("agent", None, 97)
    # `Actor` states the rule as truthiness; the schema has to state the same
    # predicate, or the empty string walks through the nullability version of it.
    with pytest.raises(IntegrityError):
        forge("agent", "", 96)
    with pytest.raises(IntegrityError):
        forge("operator", "", 95)


def test_causation_resolves_to_the_causing_event(database: Database) -> None:
    """A caused event resolves to the one that caused it, and a root to nothing.

    `causation_id` is the id of a single event, not the correlation group: the
    question "what was this a reply to" has one answer, and answering it by
    filtering a shared group id would return the whole exchange.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        asked = append_event(
            conn,
            conversation.id,
            actor=OPERATOR,
            kind="message",
            body="build it",
            correlation_id="build-1",
        )
        reported = append_event(
            conn,
            conversation.id,
            actor=BUILDER,
            kind="report",
            body="built",
            correlation_id="build-1",
            causation_id=asked.id,
        )

    with database.transaction() as conn:
        assert causing_event(conn, reported) == asked
        assert causing_event(conn, asked) is None

    # There are no FOREIGN KEYs here, so an id that resolves to nothing is
    # reachable. It must not read as a root event: the two mean opposite things.
    dangling = replace(reported, causation_id="no-such-event")
    with database.transaction() as conn:
        with pytest.raises(LookupError, match="no-such-event"):
            causing_event(conn, dangling)

    # Causation is INSIDE one conversation. An id copied from another thread
    # resolves to a real event, which is what makes it worse than a dangling
    # one: unscoped, it would read as this conversation's cause.
    with database.transaction() as conn:
        other = create_conversation(conn)
        elsewhere = append_event(
            conn, other.id, actor=OPERATOR, kind="message", body="different thread"
        )

    crossed = replace(reported, causation_id=elsewhere.id)
    with database.transaction() as conn:
        with pytest.raises(LookupError, match=elsewhere.id):
            causing_event(conn, crossed)


def test_appending_to_an_unknown_conversation_raises(database: Database) -> None:
    """A phantom transcript is not mintable by typo.

    Same argument `causing_event` already makes for `causation_id`: there are no
    FOREIGN KEYs here, so nothing at the schema level stops a caller passing an
    id that is not a conversation's, and silently accepting one produces a
    transcript that reads back fine and belongs to no conversation.
    """
    with database.transaction() as conn:
        with pytest.raises(LookupError, match="no-such-conversation"):
            append_event(
                conn, "no-such-conversation", actor=OPERATOR, kind="message", body="hi"
            )

    with database.transaction() as conn:
        assert read_transcript(conn, "no-such-conversation") == []
