#!/usr/bin/env python
"""A conversation of typed, ordered, attributable events - and nothing else.

    python examples/chat_conversation.py

`scufris_chat` owns the conversation Scufris keeps for itself rather than
borrowing from a provider, and the point of carving it out is that this can be
DEMONSTRATED: the script opens a real database, mints a conversation, appends an
operator message and an agent report inside ONE transaction, and reads the
transcript back with each event's author still typed. It imports `scufris`
nowhere, opens no socket and talks to no model.

    1. open      - a real SQLite file under a temporary directory
    2. write     - a conversation and two events in ONE `Database.transaction()`
    3. read      - the transcript, in order, with its actors
    4. resolve   - which event caused the report
    5. deliver   - two channels each send the same event, and a REPLAY of one of
                   those deliveries changes nothing

The store takes the OPEN connection rather than the `Database`, which is what
makes step 2 one atomic thing: a real caller writes its own state change in that
same block, so the change and the event describing it commit together or not at
all. `packages/chat/src/scufris_chat/README.md` is the longer version.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Run from a checkout without installing it. Only these two members' `src` is
# needed: this script imports `scufris_chat` and `scufris_core`, never `scufris`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "chat" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "core" / "src"))

from scufris_chat import (  # noqa: E402
    Actor,
    ActorKind,
    append_event,
    causing_event,
    claim_delivery,
    confirm_delivery,
    create_conversation,
    pending_events,
    read_transcript,
)
from scufris_core import Base, Database, open_database  # noqa: E402

OPERATOR = Actor(ActorKind.OPERATOR)
BUILDER = Actor(ActorKind.AGENT, "builder")

CHANNELS = ("telegram", "web")


def deliver(database: Database, conversation_id: str, channel: str) -> list[int]:
    """One channel's whole delivery pass, returning what it actually sent.

    This is the shape every channel has: ask what it is owed, claim each one,
    send, confirm. The claim's answer is what drives the send, so a pass over a
    conversation this channel has already delivered sends nothing - the replay
    is a no-op at the storage layer rather than a rule this function keeps.

    The send sits BETWEEN two units of work, never inside one. That is the whole
    reason for two states: a transaction spanning the send would hold the claim
    unwritten while the card was posted, so a crash would lose the row that
    records it and the next pass would post a second card. Committing the claim
    first makes the crash window a `claimed` row with no confirmation, which the
    next `pending_events` hands straight back.
    """
    sent: list[int] = []
    with database.transaction() as connection:
        owed = pending_events(connection, conversation_id, channel)
    for event in owed:
        with database.transaction() as connection:
            claimed = claim_delivery(
                connection, conversation_id, channel, event.event_seq
            )
        if not claimed:
            continue
        sent.append(event.event_seq)  # the "send" - a card, a message
        with database.transaction() as connection:
            confirm_delivery(connection, conversation_id, channel, event.event_seq)
    return sent


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        database = open_database(Path(tmp))
        try:
            print(f"1. opened {database.path.name} under a temporary directory")

            # No Alembic here: the APP's schema is migrated, never created from
            # the models (see scufris/db/migrate.py). `metadata` holds exactly
            # `conversation`, `event` and `delivery`, because the only package
            # this script imports that declares rows is `scufris_chat`.
            Base.metadata.create_all(database.engine)

            with database.transaction() as connection:
                conversation = create_conversation(connection)
                asked = append_event(
                    connection,
                    conversation.id,
                    actor=OPERATOR,
                    kind="message",
                    body="rebuild the dashboard",
                )
                append_event(
                    connection,
                    conversation.id,
                    actor=BUILDER,
                    kind="report",
                    body="rebuilt it; two files changed",
                    causation_id=asked.id,
                )
            print("2. wrote a conversation and two events in one transaction")

            with database.transaction() as connection:
                transcript = read_transcript(connection, conversation.id)
                print("3. transcript:")
                for event in transcript:
                    actor = event.actor.kind.value
                    if event.actor.agent_id is not None:
                        actor = f"{actor}/{event.actor.agent_id}"
                    print(
                        f"     {event.event_seq}. {actor:16} "
                        f"{event.kind:8} {event.body}"
                    )
                cause = causing_event(connection, transcript[-1])

            print(f"4. the report answers event {cause.event_seq if cause else None}")

            print("5. delivery:")
            first = {
                channel: deliver(database, conversation.id, channel)
                for channel in CHANNELS
            }
            for channel, sent in first.items():
                print(f"     {channel:16} sent events {sent}")
            replayed = {
                channel: deliver(database, conversation.id, channel)
                for channel in CHANNELS
            }
            for channel, sent in replayed.items():
                print(f"     {channel:16} replayed, sent {sent or 'nothing'}")

            if [event.event_seq for event in transcript] != [1, 2]:
                print(f"FAILED: expected events 1 and 2, got {transcript}")
                return 1
            if cause != asked:
                print(f"FAILED: the report should answer {asked}, got {cause}")
                return 1
            # Every channel is owed the WHOLE conversation independently: one
            # channel confirming says nothing about another.
            if any(sent != [1, 2] for sent in first.values()):
                print(f"FAILED: each channel should send events 1 and 2, got {first}")
                return 1
            # The replay is the claim this table exists for. A second pass must
            # send nothing at all, or a restart mid-delivery posts a second card.
            if any(sent for sent in replayed.values()):
                print(f"FAILED: a replay should send nothing, got {replayed}")
                return 1
        finally:
            database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
