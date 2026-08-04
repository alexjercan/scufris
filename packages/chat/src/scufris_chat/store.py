"""Reading and writing the conversation, over a connection someone else opened.

Functions, not a class, and every one of them takes an OPEN
``sqlalchemy.Connection`` as its first argument. There is no ``Database`` on this
surface and nothing here opens a transaction, which is what makes the invariant
from ``tasks/20260729-220835/DECISION.md`` section 4 - the state change and its
event commit TOGETHER - structural instead of a rule callers are asked to keep.
A caller that wants an event written alongside something else writes both inside
its own ``Database.transaction()``; there is no way to write one without the
other having a chance to roll it back.

``event_seq`` is read and assigned inside that same connection as
``COALESCE(MAX(event_seq), 0) + 1`` scoped to the conversation - the pattern
``HostActionRow.seq`` already uses. The begin is ``BEGIN IMMEDIATE``, so two
writers cannot both read the same maximum, and an aborted unit of work takes its
candidate number down with it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    and_,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .actors import Actor, ActorKind
from .models import ConversationRow, DeliveryRow, DeliveryState, EventRow


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """One conversation. It carries no backend; see DECISION.md section 3."""

    id: str
    created_at: float


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One attributable utterance, with its author already typed.

    ``kind`` is a plain string here as it is in the column: nothing in this
    package branches on it, and the enum lands with the first caller that does.
    """

    id: str
    conversation_id: str
    event_seq: int
    actor: Actor
    kind: str
    body: str
    correlation_id: str | None
    causation_id: str | None
    created_at: float


def create_conversation(conn: Connection) -> ConversationRecord:
    """Mint a conversation inside the caller's transaction."""
    record = ConversationRecord(id=uuid.uuid4().hex, created_at=time.time())
    conn.execute(
        insert(ConversationRow).values(id=record.id, created_at=record.created_at)
    )
    return record


def append_event(
    conn: Connection,
    conversation_id: str,
    *,
    actor: Actor,
    kind: str,
    body: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> EventRecord:
    """Append one utterance, numbering it inside the caller's transaction.

    An unknown ``conversation_id`` raises rather than minting a transcript that
    belongs to no conversation. There are no FOREIGN KEYs here, so the check is
    this function's to make; it is the same one ``causing_event`` makes for
    ``causation_id``, inside the caller's unit of work so a conversation created
    in it is visible and one created after it is not.
    """
    exists = conn.execute(
        select(ConversationRow.id).where(ConversationRow.id == conversation_id)
    ).first()
    if exists is None:
        raise LookupError(
            f"there is no conversation {conversation_id!r} to append an event to"
        )
    next_seq = (
        conn.execute(
            select(func.coalesce(func.max(EventRow.event_seq), 0)).where(
                EventRow.conversation_id == conversation_id
            )
        ).scalar_one()
        + 1
    )
    record = EventRecord(
        id=uuid.uuid4().hex,
        conversation_id=conversation_id,
        event_seq=next_seq,
        actor=actor,
        kind=kind,
        body=body,
        correlation_id=correlation_id,
        causation_id=causation_id,
        created_at=time.time(),
    )
    conn.execute(
        insert(EventRow).values(
            id=record.id,
            conversation_id=record.conversation_id,
            event_seq=record.event_seq,
            actor_kind=record.actor.kind.value,
            actor_agent_id=record.actor.agent_id,
            kind=record.kind,
            body=record.body,
            correlation_id=record.correlation_id,
            causation_id=record.causation_id,
            created_at=record.created_at,
        )
    )
    return record


def read_transcript(conn: Connection, conversation_id: str) -> list[EventRecord]:
    """The whole conversation in order. Oldest first - it is read as a script."""
    rows = conn.execute(
        select(EventRow.__table__)
        .where(EventRow.conversation_id == conversation_id)
        .order_by(EventRow.event_seq)
    ).all()
    return [_record(row) for row in rows]


def causing_event(conn: Connection, event: EventRecord) -> EventRecord | None:
    """What this event was a reply to, or ``None`` if it started something.

    ``causation_id`` names ONE event, not the correlation group: "what was this a
    reply to" has a single answer, and filtering by ``correlation_id`` would
    return the whole exchange.

    A causation id that resolves to nothing raises rather than reading as a root
    event. There are no FOREIGN KEYs here - ``scufris/db/models.py`` records why
    - so nothing at the schema level stops a caller passing an id that is not an
    event's, and the two cases mean opposite things to a reader.

    The lookup is scoped to the event's own conversation. Causation is a claim
    about the transcript this event is IN, so an id copied from another thread is
    the same error as one that resolves to nothing - and worse unscoped, because
    it would resolve to a real event and read as this conversation's cause.
    """
    if event.causation_id is None:
        return None
    row = conn.execute(
        select(EventRow.__table__).where(
            EventRow.id == event.causation_id,
            EventRow.conversation_id == event.conversation_id,
        )
    ).first()
    if row is None:
        raise LookupError(
            f"event {event.id} was caused by {event.causation_id}, which is not "
            f"an event in conversation {event.conversation_id}"
        )
    return _record(row)


def claim_delivery(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> bool:
    """Take responsibility for sending one event to one channel, once.

    ``True`` when the caller should send. That is BOTH the case where this
    attempt minted the row and the case where it found a ``claimed`` row nobody
    ever confirmed - which is exactly "someone was mid-send when we died", the
    one case a restart must retry. ``False`` only for a ``confirmed`` row, which
    is what makes a replay of a completed delivery a no-op at the STORAGE layer.
    Every channel gets that guarantee without implementing it, and a channel
    added later cannot forget it.

    The key is derived from the event, so a retry after a crash recomputes the
    same three values and finds its own row rather than writing a second one.
    The existing row is looked for BEFORE the event is required, so a replay of
    an already-confirmed delivery stays a no-op rather than becoming a
    ``LookupError`` once retention starts removing the events underneath it.

    The INSERT resolves its own conflict rather than trusting the read that
    preceded it. The caller's begin is immediate
    (``scufris_core.engine``), so two claimants cannot in fact both see nothing
    there - but a connection from any other engine would then turn the loser
    into an ``IntegrityError`` instead of a re-claim, and this way the answer is
    true by construction.
    """
    state = _delivery_state(conn, conversation_id, channel, event_seq)
    if state is None:
        _require_event(conn, conversation_id, channel, event_seq)
        minted = conn.execute(
            sqlite_insert(DeliveryRow)
            .values(
                channel=channel,
                conversation_id=conversation_id,
                event_seq=event_seq,
                state=DeliveryState.CLAIMED.value,
                claimed_at=time.time(),
                confirmed_at=None,
            )
            .on_conflict_do_nothing()
        )
        if minted.rowcount:
            return True
        # Another claimant got there between the read and the insert. Its row is
        # the one that counts, and it is answered the same way any other
        # pre-existing row is.
        state = conn.execute(
            select(DeliveryRow.state).where(
                *_delivery_key(conversation_id, channel, event_seq)
            )
        ).scalar_one()
    if state == DeliveryState.CONFIRMED.value:
        return False
    # A claimed row nobody confirmed. Hand it back to be sent again, stamped
    # with this attempt's time rather than the dead one's.
    conn.execute(
        update(DeliveryRow)
        .where(*_delivery_key(conversation_id, channel, event_seq))
        .values(claimed_at=time.time())
    )
    return True


def confirm_delivery(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> None:
    """Record that the channel's send returned. Call it after the send, not before.

    Until this lands the delivery stays pending, so a crash between the send and
    this call is retried rather than lost - a duplicate card, not a question the
    operator never sees.

    A delivery that is not sitting in ``claimed`` raises rather than passing
    silently, for the same reason ``append_event`` refuses an unknown
    conversation: there are no FOREIGN KEYs here, and a silent no-op would read
    as a delivery that completed. No correct caller reaches it - every one gates
    its send behind a ``True`` from ``claim_delivery``, which hands back only a
    row it left ``claimed``.
    """
    confirmed = conn.execute(
        update(DeliveryRow)
        .where(
            *_delivery_key(conversation_id, channel, event_seq),
            DeliveryRow.state == DeliveryState.CLAIMED.value,
        )
        .values(state=DeliveryState.CONFIRMED.value, confirmed_at=time.time())
    )
    if not confirmed.rowcount:
        raise LookupError(
            f"channel {channel!r} has no claimed delivery of event {event_seq} "
            f"of conversation {conversation_id!r} to confirm"
        )


def pending_events(
    conn: Connection, conversation_id: str, channel: str
) -> list[EventRecord]:
    """What this channel should send now, oldest first.

    Events with no delivery row for the channel, PLUS events whose row was
    claimed and never confirmed. One function rather than two: every caller
    wants the union, and two names would invite a channel to ask one and forget
    the other, which is the per-channel forgetting this table exists to prevent.

    It answers for a channel that did not exist when the events were written,
    because nothing declares the set of channels and no rows are fanned out at
    append time - the absence of a row IS the backlog.

    Whether a long-offline channel then sends all of these or only the ones
    still unresolved is the caller's predicate over this result, not a shape
    decided here.
    """
    outstanding = EventRow.__table__.outerjoin(
        DeliveryRow.__table__,
        and_(
            DeliveryRow.channel == channel,
            DeliveryRow.conversation_id == EventRow.conversation_id,
            DeliveryRow.event_seq == EventRow.event_seq,
        ),
    )
    rows = conn.execute(
        select(EventRow.__table__)
        .select_from(outstanding)
        .where(
            EventRow.conversation_id == conversation_id,
            or_(
                DeliveryRow.channel.is_(None),
                DeliveryRow.state == DeliveryState.CLAIMED.value,
            ),
        )
        .order_by(EventRow.event_seq)
    ).all()
    return [_record(row) for row in rows]


def _delivery_key(
    conversation_id: str, channel: str, event_seq: int
) -> tuple[ColumnElement[bool], ...]:
    """The three primary-key criteria, in one place rather than four."""
    return (
        DeliveryRow.channel == channel,
        DeliveryRow.conversation_id == conversation_id,
        DeliveryRow.event_seq == event_seq,
    )


def _delivery_state(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> str | None:
    """This channel's state for this event, or ``None`` if it has never tried."""
    return conn.execute(
        select(DeliveryRow.state).where(
            *_delivery_key(conversation_id, channel, event_seq)
        )
    ).scalar_one_or_none()


def _require_event(
    conn: Connection, conversation_id: str, channel: str, event_seq: int
) -> None:
    """Refuse a delivery of something that was never said.

    There are no FOREIGN KEYs here, so this is the store's check to make - the
    same one ``append_event`` makes for ``conversation_id``. Inside the caller's
    unit of work, so an event appended in that same block is claimable and one
    appended after it is not.
    """
    exists = conn.execute(
        select(EventRow.id).where(
            EventRow.conversation_id == conversation_id,
            EventRow.event_seq == event_seq,
        )
    ).first()
    if exists is None:
        raise LookupError(
            f"channel {channel!r} cannot deliver event {event_seq} of "
            f"conversation {conversation_id!r}: there is no such event"
        )


def _record(row: Row[tuple[object, ...]]) -> EventRecord:
    return EventRecord(
        id=row.id,
        conversation_id=row.conversation_id,
        event_seq=row.event_seq,
        actor=Actor(ActorKind(row.actor_kind), row.actor_agent_id),
        kind=row.kind,
        body=row.body,
        correlation_id=row.correlation_id,
        causation_id=row.causation_id,
        created_at=row.created_at,
    )
