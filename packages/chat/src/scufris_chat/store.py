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

from sqlalchemy import Connection, Row, func, insert, select

from .actors import Actor, ActorKind
from .models import ConversationRow, EventRow


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
