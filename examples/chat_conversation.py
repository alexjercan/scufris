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
    6. switch    - the provider session is a CACHE: binding one backend hits,
                   the other misses, and re-seeding from assembled context
                   leaves the conversation byte-identical

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
    CONTEXT_POLICY_VERSION,
    Actor,
    ActorKind,
    EventRecord,
    append_event,
    assemble_context,
    bind_session,
    cached_session,
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


def print_transcript(events: list[EventRecord]) -> None:
    """One line per event, numbered and attributed.

    Shared by step 3 and step 6 so the before and after of a backend switch are
    printed the same way: the claim is that the two are identical, and two
    formatters would let a difference in the rendering read as a difference in
    the conversation.
    """
    for event in events:
        actor = event.actor.kind.value
        if event.actor.agent_id is not None:
            actor = f"{actor}/{event.actor.agent_id}"
        print(f"     {event.event_seq}. {actor:16} {event.kind:8} {event.body}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        database = open_database(Path(tmp))
        try:
            print(f"1. opened {database.path.name} under a temporary directory")

            # No Alembic here: the APP's schema is migrated, never created from
            # the models (see scufris/db/migrate.py). `metadata` holds exactly
            # `conversation`, `event`, `delivery` and `provider_session`, because
            # the only package this script imports that declares rows is
            # `scufris_chat`.
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
                print_transcript(transcript)
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

            print("6. the provider session is a cache:")
            with database.transaction() as connection:
                codex = bind_session(
                    connection,
                    conversation.id,
                    backend="codex",
                    policy_version=CONTEXT_POLICY_VERSION,
                    provider_session_id="rollout_xyz",
                )
                hit = cached_session(
                    connection,
                    conversation.id,
                    backend="codex",
                    policy_version=CONTEXT_POLICY_VERSION,
                )
                # The switch itself writes NOTHING. It is "use claude next turn",
                # and the miss below is what drives the re-seed - the same miss a
                # restart or a provider-side compaction produces, with no switch
                # event to hang eager work on.
                miss = cached_session(
                    connection,
                    conversation.id,
                    backend="claude",
                    policy_version=CONTEXT_POLICY_VERSION,
                )
            print(f"     codex   -> {hit.provider_session_id if hit else 'MISS'}")
            print(f"     claude  -> {miss.provider_session_id if miss else 'MISS'}")

            with database.transaction() as connection:
                seed = assemble_context(connection, conversation.id)
                claude = bind_session(
                    connection,
                    conversation.id,
                    backend="claude",
                    policy_version=CONTEXT_POLICY_VERSION,
                    provider_session_id="sess_abc",
                )
            print("     assembled context, every line attributed:")
            for line in seed.splitlines():
                print(f"       {line}")
            print(f"     claude  -> {claude.provider_session_id} (re-seeded)")

            with database.transaction() as connection:
                after = read_transcript(connection, conversation.id)
            print("     the transcript after the switch:")
            print_transcript(after)

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
            # The release promise, checked rather than described: the backend
            # changed underneath the conversation and the conversation did not
            # notice. Both halves - an unchanged transcript alone would also be
            # true of a switch that never happened.
            if after != transcript:
                print(f"FAILED: the switch changed the conversation: {after}")
                return 1
            if codex.provider_session_id == claude.provider_session_id:
                print("FAILED: the two backends should hold different sessions")
                return 1
            if miss is not None:
                print(f"FAILED: the new backend should miss the cache, got {miss}")
                return 1
            # The other half of step 6: a warm lookup has to hand back what was
            # bound, or every turn re-seeds and the cache is a write-only table.
            if hit is None or hit.provider_session_id != codex.provider_session_id:
                print(f"FAILED: the warm lookup should return {codex}, got {hit}")
                return 1
        finally:
            database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
