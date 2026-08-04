"""The six claims the `delivery` table is for.

Each one is a property of the DATA rather than of a channel's care, which is the
whole reason the table exists: the key is DERIVED from the event, so a retry
after a crash recomputes the same three values and collides instead of producing
a second card, and a channel added later inherits that without implementing it.

`tasks/20260804-115319/DECISION.md` is what these assert.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import text

from scufris_chat import (
    Actor,
    ActorKind,
    append_event,
    claim_delivery,
    confirm_delivery,
    create_conversation,
    pending_events,
)
from scufris_core import Base, Database, open_database

#: The tables this package owns, created straight from the shared metadata.
OWNED_TABLES = ("conversation", "event", "delivery")

OPERATOR = Actor(ActorKind.OPERATOR)

TELEGRAM = "telegram"
WEB = "web"


class Boom(RuntimeError):
    """Raised inside a transaction to make the rollback real."""


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """A file-backed state database holding this package's tables.

    Owned here rather than borrowed, and created from the metadata rather than
    by running Alembic, for the reason `test_chat_events.py` records: the
    migration environment ships with the ROOT distribution.
    `tests/test_db_schema.py::test_migration_creates_the_delivery_table` is what
    proves the shipped migration agrees with this metadata.
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


def _delivery_rows(database: Database) -> list[tuple[str, str, int, str]]:
    """Every delivery row as `(channel, conversation_id, event_seq, state)`.

    Read with raw SQL on purpose: the store exports no delivery reader beyond
    `pending_events`, and the claims here are about what is ON DISK - one row
    rather than two, nothing at all after a rollback - which a function that
    filters by what is still pending cannot express.
    """
    with database.transaction() as conn:
        rows = conn.execute(
            text(
                "SELECT channel, conversation_id, event_seq, state FROM delivery "
                "ORDER BY channel, conversation_id, event_seq"
            )
        ).all()
    return [(row[0], row[1], row[2], row[3]) for row in rows]


def test_delivery_is_idempotent_on_replay(database: Database) -> None:
    """The same event sent to the same channel twice is one row and one send.

    The counter is what makes this a claim about the SIDE EFFECT rather than
    about the table: a channel drives its send off the claim's answer, so a
    second attempt that returned `True` would post a second card even with the
    row count still at one.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        event = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="approve?"
        )

    sent: list[str] = []

    def deliver() -> None:
        with database.transaction() as conn:
            if claim_delivery(conn, conversation.id, TELEGRAM, event.event_seq):
                sent.append(event.id)
                confirm_delivery(conn, conversation.id, TELEGRAM, event.event_seq)

    deliver()
    deliver()
    deliver()

    assert sent == [event.id]
    assert _delivery_rows(database) == [
        (TELEGRAM, conversation.id, event.event_seq, "confirmed")
    ]
    with database.transaction() as conn:
        assert pending_events(conn, conversation.id, TELEGRAM) == []


def test_one_event_reaches_every_channel(database: Database) -> None:
    """One event, two channels, two independent deliveries.

    The key's leading column is the channel, so one channel confirming says
    nothing about another - which is what makes "delivered" a per-channel fact
    rather than a single flag the second channel would find already set.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        event = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="approve?"
        )

    with database.transaction() as conn:
        assert claim_delivery(conn, conversation.id, TELEGRAM, event.event_seq) is True
        confirm_delivery(conn, conversation.id, TELEGRAM, event.event_seq)

    # The other channel has not been told, and the first one's confirmation is
    # not an answer on its behalf.
    with database.transaction() as conn:
        assert [
            pending.id for pending in pending_events(conn, conversation.id, WEB)
        ] == [event.id]
        assert claim_delivery(conn, conversation.id, WEB, event.event_seq) is True
        confirm_delivery(conn, conversation.id, WEB, event.event_seq)

    assert _delivery_rows(database) == [
        (TELEGRAM, conversation.id, event.event_seq, "confirmed"),
        (WEB, conversation.id, event.event_seq, "confirmed"),
    ]


def test_event_and_delivery_commit_atomically(database: Database) -> None:
    """A unit of work that unwinds leaves neither the event nor its delivery.

    Both are written through the caller's OPEN connection and neither function
    opens one, so this is structural rather than a rule a caller is asked to
    keep. A store that opened its own transaction to write the claim would leave
    a delivery row pointing at an event that was never committed - and that row
    would then suppress the delivery of the event's eventual replacement.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)

    with pytest.raises(Boom):
        with database.transaction() as conn:
            event = append_event(
                conn, conversation.id, actor=OPERATOR, kind="message", body="approve?"
            )
            claim_delivery(conn, conversation.id, TELEGRAM, event.event_seq)
            raise Boom("the channel's caller failed after claiming")

    assert _delivery_rows(database) == []
    with database.transaction() as conn:
        assert pending_events(conn, conversation.id, TELEGRAM) == []


def test_delivery_requires_its_event(database: Database) -> None:
    """A delivery of something that was never said cannot be written.

    There are no FOREIGN KEYs here, so this is the store's check to make - the
    same one `append_event` makes for `conversation_id`. Made INSIDE the
    caller's unit of work, so an event appended in that same block is visible to
    the claim that follows it.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="approve?"
        )

    with database.transaction() as conn:
        with pytest.raises(LookupError, match=TELEGRAM):
            claim_delivery(conn, conversation.id, TELEGRAM, 99)
        with pytest.raises(LookupError, match="no-such-conversation"):
            claim_delivery(conn, "no-such-conversation", TELEGRAM, 1)

    # Confirming what was never claimed is the same class of error: it would
    # otherwise be a silent no-op that reads as a successful delivery.
    with database.transaction() as conn:
        with pytest.raises(LookupError, match=TELEGRAM):
            confirm_delivery(conn, conversation.id, TELEGRAM, 1)

    assert _delivery_rows(database) == []

    # An event appended in the SAME unit of work is there to be claimed, which
    # is the case a check made outside the caller's transaction would refuse.
    with database.transaction() as conn:
        second = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="and this?"
        )
        assert claim_delivery(conn, conversation.id, TELEGRAM, second.event_seq) is True


def test_a_claimed_delivery_is_pending_until_confirmed(database: Database) -> None:
    """A crash between the send and the confirm is retried, not lost.

    The two states are what make the crash window READABLE: a `claimed` row with
    no confirmation is exactly "someone was mid-send when we died", and it stays
    pending so a restart sends it again. One boolean row would have to be
    written either before the send - and silently lose the question - or after
    it, and duplicate forever.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        event = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="approve?"
        )

    sent: list[int] = []

    def restart() -> None:
        """The whole pass a channel runs on startup, claim included.

        Driving the SAME loop the README and `examples/chat_conversation.py`
        document is the point: a recovery half that called `confirm_delivery`
        directly would prove a claimed row is pending while leaving the path an
        actual restart takes - `pending_events`, then `claim_delivery`, whose
        answer gates the send - untested. A claim that refused its own abandoned
        row would strand it pending forever, which is the "silently loses the
        question" failure the two states exist to rule out.
        """
        with database.transaction() as conn:
            owed = pending_events(conn, conversation.id, TELEGRAM)
        for pending in owed:
            with database.transaction() as conn:
                claimed = claim_delivery(
                    conn, conversation.id, TELEGRAM, pending.event_seq
                )
            if not claimed:
                continue
            sent.append(pending.event_seq)
            with database.transaction() as conn:
                confirm_delivery(conn, conversation.id, TELEGRAM, pending.event_seq)

    # The claim commits, and then the process dies before the send returns.
    with database.transaction() as conn:
        assert claim_delivery(conn, conversation.id, TELEGRAM, event.event_seq) is True
    assert _delivery_rows(database) == [
        (TELEGRAM, conversation.id, event.event_seq, "claimed")
    ]

    with database.transaction() as conn:
        assert [
            pending.id for pending in pending_events(conn, conversation.id, TELEGRAM)
        ] == [event.id]

    # The restart sends it - once - and the row ends confirmed.
    restart()

    assert sent == [event.event_seq]
    assert _delivery_rows(database) == [
        (TELEGRAM, conversation.id, event.event_seq, "confirmed")
    ]
    with database.transaction() as conn:
        assert pending_events(conn, conversation.id, TELEGRAM) == []

    # And the NEXT restart is the replay case again: nothing owed, nothing sent.
    restart()

    assert sent == [event.event_seq]


def test_a_channel_that_was_offline_sees_what_it_missed(database: Database) -> None:
    """A channel is owed every event it has no confirmation for, in order.

    Including a channel that did not EXIST when the events were written, which
    is the case a fan-out of delivery rows at append time cannot answer without
    knowing every future channel's name. Nothing declares the set of channels; a
    row appears only where one attempted.
    """
    with database.transaction() as conn:
        conversation = create_conversation(conn)
        first = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="one"
        )
        second = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="two"
        )
        third = append_event(
            conn, conversation.id, actor=OPERATOR, kind="message", body="three"
        )

    for event in (first, second, third):
        with database.transaction() as conn:
            assert claim_delivery(conn, conversation.id, TELEGRAM, event.event_seq)
            confirm_delivery(conn, conversation.id, TELEGRAM, event.event_seq)

    with database.transaction() as conn:
        assert pending_events(conn, conversation.id, TELEGRAM) == []
        # A channel with no rows at all is owed everything, oldest first: it is
        # read as a transcript, so the order is the claim as much as the set.
        assert [
            pending.event_seq for pending in pending_events(conn, conversation.id, WEB)
        ] == [
            first.event_seq,
            second.event_seq,
            third.event_seq,
        ]

    # The union is ONE read: a channel part-way through owes both what it never
    # claimed and what it claimed and never confirmed.
    with database.transaction() as conn:
        assert claim_delivery(conn, conversation.id, WEB, first.event_seq) is True
        confirm_delivery(conn, conversation.id, WEB, first.event_seq)
        assert claim_delivery(conn, conversation.id, WEB, second.event_seq) is True

    with database.transaction() as conn:
        assert [
            pending.event_seq for pending in pending_events(conn, conversation.id, WEB)
        ] == [
            second.event_seq,
            third.event_seq,
        ]

    # Scoped to its conversation: another thread's events are not this one's
    # backlog, however long the channel was away.
    with database.transaction() as conn:
        other = create_conversation(conn)
        append_event(conn, other.id, actor=OPERATOR, kind="message", body="elsewhere")

    with database.transaction() as conn:
        assert [
            pending.event_seq for pending in pending_events(conn, conversation.id, WEB)
        ] == [
            second.event_seq,
            third.event_seq,
        ]
