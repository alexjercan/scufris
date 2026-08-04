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
    create_conversation,
    read_transcript,
)
from scufris_core import Base, open_database  # noqa: E402

OPERATOR = Actor(ActorKind.OPERATOR)
BUILDER = Actor(ActorKind.AGENT, "builder")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        database = open_database(Path(tmp))
        try:
            print(f"1. opened {database.path.name} under a temporary directory")

            # No Alembic here: the APP's schema is migrated, never created from
            # the models (see scufris/db/migrate.py). `metadata` holds exactly
            # `conversation` and `event`, because the only package this script
            # imports that declares rows is `scufris_chat`.
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

            if [event.event_seq for event in transcript] != [1, 2]:
                print(f"FAILED: expected events 1 and 2, got {transcript}")
                return 1
            if cause != asked:
                print(f"FAILED: the report should answer {asked}, got {cause}")
                return 1
        finally:
            database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
