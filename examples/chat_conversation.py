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
    3. authorize - the operator's message mints a decision; the agent's report
                   is refused, and a system notice records which event decided
    4. read      - the transcript as a tree: ordered, one colour per actor kind,
                   every answer drawn under the event it answers
    5. deliver   - two channels each send the same events, and a REPLAY of one
                   of those deliveries changes nothing
    6. switch    - the provider session is a CACHE: binding one backend hits,
                   the other misses, and re-seeding from assembled context
                   leaves the conversation byte-identical

The store takes the OPEN connection rather than the `Database`, which is what
makes step 2 one atomic thing: a real caller writes its own state change in that
same block, so the change and the event describing it commit together or not at
all. `packages/chat/src/scufris_chat/README.md` is the longer version.

The one import from outside the two members is `rich`, which draws the tree in
step 4. It is a dependency of the repository's dev shell (the root
`pyproject.toml`), not of `scufris-chat`, so this script runs from a checkout
rather than from an install of the package it demonstrates.

Every line of output above is a CHECKED claim. The tree is rendered into a
recording console and asserted on before it is printed, so the string this
script vouches for and the string an operator reads are one string rather than
two renderings that can drift; the exit code is the verdict, and
`tests/test_examples.py` reads the output back as well.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

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
    authorize,
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
SYSTEM = Actor(ActorKind.SYSTEM)

CHANNELS = ("telegram", "web")

#: One colour per actor KIND, which is what makes an attribution readable at a
#: glance rather than only correct. Keyed by the enum and required to cover it
#: exactly, checked in `main`: a kind with no colour would raise here the first
#: time anything wrote one, and the coordinator's `orchestrator` is exactly that
#: case waiting to happen.
ACTOR_STYLES: dict[ActorKind, str] = {
    ActorKind.OPERATOR: "bold cyan",
    ActorKind.ORCHESTRATOR: "bold magenta",
    ActorKind.AGENT: "bold green",
    ActorKind.SYSTEM: "dim white",
}

#: The edges `rich.tree.Tree` draws between an event and the one it answers.
TREE_GUIDES = ("├── ", "└── ")

#: Everything a guide is made of, plus the space the tree indents with. Stripped
#: off the left of a rendered line to recover the event it carries.
GUIDE_CHARACTERS = " │├└─"

#: Fixed rather than taken from the terminal, so the transcript the script
#: asserts on is the transcript every reader sees, at any window size.
CONSOLE_WIDTH = 100


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


def read_transcript_of(database: Database, conversation_id: str) -> list[EventRecord]:
    """The whole thread, in its own unit of work.

    A one-line wrapper because the transcript is read twice - once before the
    backend switch and once after - and the claim is that the two are identical.
    Two open-coded reads would be two places for that to stop being true.
    """
    with database.transaction() as connection:
        return read_transcript(connection, conversation_id)


def causation(database: Database, events: list[EventRecord]) -> dict[int, int]:
    """Which event each of these answers, as sequence numbers.

    `causing_event` resolves one `causation_id` against THIS conversation, so
    what comes back is the transcript's own edge rather than a number the caller
    could have made up. Absent for an event that started something.
    """
    causes: dict[int, int] = {}
    with database.transaction() as connection:
        for event in events:
            cause = causing_event(connection, event)
            if cause is not None:
                causes[event.event_seq] = cause.event_seq
    return causes


def depth(line: str) -> int:
    """How deep in the tree a rendered line sits."""
    return len(line) - len(line.lstrip(GUIDE_CHARACTERS))


def parent_line(lines: list[str], line: str) -> str | None:
    """The line `line` is drawn UNDER, or None if it hangs off the root.

    The nearest preceding line drawn shallower than this one, which is what
    "under" means in a rendered tree. A depth comparison alone does not say it:
    a line drawn under the WRONG ancestor is still deeper than the event it
    should have answered, so `depth(answer) > depth(asked)` stays true of a
    transcript that redrew its causation.

    The first shallower line decides, even when it is the root - walking past
    it would leave the tree and start reading the prose above it as ancestry.
    """
    above = lines[: lines.index(line)]
    for candidate in reversed(above):
        if depth(candidate) < depth(line):
            return candidate if depth(candidate) > 0 else None
    return None


def render_transcript(events: list[EventRecord], causes: dict[int, int]) -> str:
    """The transcript as a causation tree: ordered, attributed, and RETURNED.

    Returned rather than printed. The caller asserts on the exact string it is
    about to show, so the text this script vouches for and the text an operator
    reads cannot differ; a renderer that printed would leave the assertion to
    re-render its own copy.

    The console records into a buffer of its own for the same reason. Colour is
    real - `ACTOR_STYLES` is applied per event and a terminal shows it - but a
    piped run emits none, so what `export_text` returns is what a piped reader
    gets, which is what the gate reads back.

    Shared by step 4 and step 6, so the before and after of a backend switch are
    drawn the same way: the claim is that the two are identical, and two
    renderers would let a difference in the drawing read as a difference in the
    conversation.
    """
    console = Console(record=True, width=CONSOLE_WIDTH, file=io.StringIO())
    tree = Tree("conversation")
    nodes: dict[int, Tree] = {}
    for event in events:
        label = Text(
            f"{event.event_seq}. {event.actor.render():16} {event.kind:8} {event.body}",
            style=ACTOR_STYLES[event.actor.kind],
        )
        parent = nodes.get(causes.get(event.event_seq, 0), tree)
        nodes[event.event_seq] = parent.add(label)
    console.print(tree)
    return console.export_text()


def tree_problems(
    text: str, events: list[EventRecord], causes: dict[int, int]
) -> list[str]:
    """Every claim the rendered tree makes and fails to keep, as messages.

    All of them rather than the first, so one run says everything that is wrong
    with what it drew. A rendering nobody checks is decoration: this script's
    gate is its exit code, and a tree that lost its attribution, its order or
    its edges would still exit 0 without this.
    """
    problems: list[str] = []
    lines = text.splitlines()
    drawn: dict[int, str] = {}
    for event in events:
        matches = [
            line
            for line in lines
            if line.lstrip(GUIDE_CHARACTERS).startswith(f"{event.event_seq}. ")
        ]
        if len(matches) != 1:
            problems.append(
                f"event {event.event_seq} is drawn {len(matches)} times, not once"
            )
            continue
        drawn[event.event_seq] = matches[0]
        if event.actor.render() not in matches[0]:
            problems.append(
                f"event {event.event_seq} is drawn without naming its author, "
                f"{event.actor.render()}"
            )
    for event_seq, cause_seq in sorted(causes.items()):
        answer, asked = drawn.get(event_seq), drawn.get(cause_seq)
        if answer is None or asked is None:
            continue
        if parent_line(lines, answer) != asked:
            problems.append(
                f"event {event_seq} answers event {cause_seq} and is drawn "
                "beside it or under something else rather than under it"
            )
        if lines.index(answer) < lines.index(asked):
            problems.append(
                f"event {event_seq} is drawn before event {cause_seq}, which it answers"
            )
    if not any(guide in text for guide in TREE_GUIDES):
        problems.append("the tree draws no edges at all, so it renders no causation")
    return problems


def main() -> int:
    # Before anything is rendered: a kind with no colour is a `KeyError` in
    # `render_transcript` the first time one is written, and `orchestrator` -
    # which nothing writes until the coordinator lands - is that case waiting.
    if set(ACTOR_STYLES) != set(ActorKind):
        print(
            f"FAILED: a colour per typed actor means all of "
            f"{[kind.value for kind in ActorKind]}, got "
            f"{sorted(kind.value for kind in ACTOR_STYLES)}"
        )
        return 1

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
                reported = append_event(
                    connection,
                    conversation.id,
                    actor=BUILDER,
                    kind="report",
                    body="rebuilt it; two files changed",
                    causation_id=asked.id,
                )
            print("2. wrote a conversation and two events in one transaction")

            # The lane's last claim: a stop gate takes an `OperatorDecision`, so
            # a caller holding none cannot phrase the call at all, and
            # `authorize` is its only mint. Both halves are shown - the mint from
            # the operator's message, and the refusal of the agent's report,
            # which is an untrusted quotation no matter how it is phrased.
            with database.transaction() as connection:
                decision = authorize(connection, conversation.id, asked.event_seq)
                try:
                    authorize(connection, conversation.id, reported.event_seq)
                except PermissionError as refusal:
                    refused = str(refusal)
                else:
                    refused = ""
                append_event(
                    connection,
                    conversation.id,
                    actor=SYSTEM,
                    kind="notice",
                    body=(
                        f"event {decision.event_seq} authorized the rebuild; "
                        "the agent's report did not"
                    ),
                    causation_id=asked.id,
                )
            print("3. only an operator event authorizes:")
            print(
                f"     event {decision.event_seq} minted a decision for "
                f"{decision.actor.render()}"
            )
            print(f"     event {reported.event_seq} was refused: {refused}")

            transcript = read_transcript_of(database, conversation.id)
            causes = causation(database, transcript)
            rendered = render_transcript(transcript, causes)
            problems = tree_problems(rendered, transcript, causes)
            if problems:
                for problem in problems:
                    print(f"FAILED: {problem}")
                print(rendered, end="")
                return 1
            print("4. the transcript, every answer under what it answers:")
            print(rendered, end="")

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

            after = read_transcript_of(database, conversation.id)
            after_causes = causation(database, after)
            after_rendered = render_transcript(after, after_causes)
            print("     the transcript after the switch, by the same renderer:")
            print(after_rendered, end="")

            if [event.event_seq for event in transcript] != [1, 2, 3]:
                print(f"FAILED: expected events 1, 2 and 3, got {transcript}")
                return 1
            # The decision is a capability, not a boolean: it names the event and
            # the actor it attests to, which is what a consumer writes into its
            # own journal.
            if decision.event_seq != asked.event_seq or decision.actor != OPERATOR:
                print(f"FAILED: the decision should attest to {asked}, got {decision}")
                return 1
            # The refusal has to NAME the actor it refused, or a caller reading
            # the error cannot tell an unauthorized event from a missing one.
            if BUILDER.render() not in refused:
                print(
                    f"FAILED: the refusal should name {BUILDER.render()}: {refused!r}"
                )
                return 1
            # Every channel is owed the WHOLE conversation independently: one
            # channel confirming says nothing about another.
            if any(sent != [1, 2, 3] for sent in first.values()):
                print(f"FAILED: each channel should send every event, got {first}")
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
            # And the same claim as the operator meets it: not merely equal
            # records, the same drawing, down to the byte.
            if after_rendered != rendered:
                print("FAILED: the switch changed what the transcript looks like")
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
