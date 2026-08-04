# Deliver chat events to every channel exactly once

- PRIORITY: 99
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256

## Story

As the operator, I want one conversation event to reach every channel exactly
once, so that a restart mid-delivery does not duplicate a card and a channel
that was offline does not silently miss the question it was supposed to ask me.

## Steps

- [x] Write `packages/chat/tests/test_chat_delivery.py` FIRST, red, over the
      `database` fixture pattern `test_chat_events.py` already uses (file-backed
      `open_database`, tables from `Base.metadata`, `OWNED_TABLES` grown to
      `("conversation", "event", "delivery")`). Six tests, named in Definition of
      Done. `pytest packages/chat/tests -q -k delivery` currently exits 5, no
      tests collected - that is the red this step turns green.
- [x] Add `DeliveryRow` to `packages/chat/src/scufris_chat/models.py`:
      `channel`, `conversation_id`, `event_seq` as a composite
      `PrimaryKeyConstraint`, plus `state`, `claimed_at`, `confirmed_at`. No
      FOREIGN KEYs, for the reason that module's docstring already records.
      The key is DERIVED from the event, not minted per attempt: that is what
      makes a retry after a crash collide rather than duplicate. Three columns
      rather than a rendered `idempotency_key` string - DECISION.md section 1.
- [x] Constrain `state` with a CHECK over the two values, rendered from the
      declaring enum the way `_ACTOR_KIND_CHECK` is rendered from
      `ACTOR_KIND_VALUES`, so a third state cannot land with the constraint
      still naming two. Pair it with a CHECK that `confirmed_at IS NOT NULL`
      exactly when `state = 'confirmed'` - the same both-halves reasoning as
      `_ACTOR_AGENT_ID_CHECK`, and for the same reason: the store rebuilds every
      row into a record.
- [x] Add three functions to `packages/chat/src/scufris_chat/store.py`, each
      taking the caller's OPEN `Connection` as its first argument, as every
      function there already does:
      - `claim_delivery(conn, channel, conversation_id, event_seq) -> bool` -
        `True` when this attempt minted the row, `False` when it was already
        there. Replay is a no-op at the STORAGE layer, so every channel gets the
        guarantee without implementing it twice and a channel added later cannot
        forget.
      - `confirm_delivery(conn, channel, conversation_id, event_seq) -> None` -
        moves a claimed row to confirmed. Called after the channel's send
        returns.
      - `pending_events(conn, conversation_id, channel) -> list[EventRecord]` -
        events with no delivery row for that channel, plus those with a claimed
        row never confirmed, in `event_seq` order. ONE read, not two -
        DECISION.md section 5.
- [x] Refuse a delivery whose event does not exist, with a `LookupError` naming
      the channel and the key. There are no FOREIGN KEYs here, so this is the
      store's check to make - the same one `append_event` makes for
      `conversation_id` and `causing_event` makes for `causation_id`, and made
      inside the caller's unit of work so an event appended in it is visible.
- [x] Export the three functions and the `DeliveryRecord` (if the shape needs
      one) from `packages/chat/src/scufris_chat/__init__.py`, and from
      `__all__`. No row class is exported; `DeliveryRow` stays private, as
      `EventRow` is.
- [x] Generate the Alembic revision with `down_revision = "18c9104709b8"` and
      confirm `test_schema_has_no_pending_autogenerate_diff` is green - that
      test is what proves the revision matches the models rather than a
      hand-edit that drifted.
- [x] Grow `tests/test_db_migrations.py`: add `"delivery"` to
      `test_declared_tables_are_the_only_ones` (and correct its docstring, which
      currently states `delivery` is deliberately absent), and add
      `test_migration_creates_the_delivery_table` asserting the composite
      primary key and both CHECKs by INSERTing against them, as
      `test_migration_creates_the_chat_tables` does - a constraint SQLite parsed
      but does not enforce would still appear in `sqlite_master`.
      Landed in a NEW `tests/test_db_schema.py`: the addition took
      `test_db_migrations.py` to 917 lines against the 900-line test cap, whose
      allowlist is a ratchet no entry may be added to. The cut is by subject -
      the runner (reaching head, the connection, the backup, where scripts ship
      from) stays, and what the revisions LEAVE BEHIND moves, which is where
      both new assertions belong anyway. The `fresh` fixture moved to
      `tests/conftest.py` rather than being duplicated across the two.
- [x] Grow `examples/chat_conversation.py` with a delivery section: two channels
      each claim and confirm the same event, then the SAME delivery is replayed
      and changes nothing, with an assertion behind it and a non-zero exit if it
      does not hold. Keep the script offline and `scufris`-free; it is gated by
      `tests/test_examples.py`.
- [x] Add a section 5 to `packages/chat/src/scufris_chat/README.md` for the
      table, the two states, and the guarantee stated honestly - exactly-once on
      the normal path, at-least-once across a crash mid-send. Link DECISION.md.
- [x] Run `pytest packages/chat/tests tests/test_db_migrations.py
      tests/test_examples.py tests/test_package_boundaries.py -q` plus
      `tatr check`.

## Definition of Done

- Delivering the same event to the same channel twice writes one row and
  performs one side effect
  (test: `test_delivery_is_idempotent_on_replay`).
- Two channels each receive the same event independently
  (test: `test_one_event_reaches_every_channel`).
- The event and its delivery are visible together or not at all
  (test: `test_event_and_delivery_commit_atomically`).
- A delivery whose event does not exist cannot be written
  (test: `test_delivery_requires_its_event`).
- A claimed delivery is still pending until it is confirmed, so a crash between
  the send and the confirm is retried rather than lost
  (test: `test_a_claimed_delivery_is_pending_until_confirmed`).
- A channel that was never told about an event sees it as pending, including a
  channel that did not exist when the event was written
  (test: `test_a_channel_that_was_offline_sees_what_it_missed`).
- The shipped migration builds the table with its composite key and both CHECKs,
  asserted by INSERTing against them
  (test: `tests/test_db_schema.py::test_migration_creates_the_delivery_table`).
- `delivery` is the only table the schema gained
  (test: `tests/test_db_schema.py::test_declared_tables_are_the_only_ones`).
- `examples/chat_conversation.py` exits 0 and prints a replayed delivery
  changing nothing (test: `tests/test_examples.py`).

  cmd: `pytest packages/chat/tests tests/test_db_migrations.py tests/test_db_schema.py tests/test_examples.py -q`
  (red on base: `pytest packages/chat/tests -q -k delivery` exits 5, no tests
  collected)

## Close-out

**What and why.** A `delivery` table keyed `(channel, conversation_id,
event_seq)`, three store functions over the caller's open connection, and the
revision that ships it. Every part of the key is derived from the event, so a
retry after a crash recomputes the same three values and collides with its own
row instead of posting a second card - idempotency became a property of the
STORAGE layer, which is what stops the next channel from reimplementing or
forgetting it. Two states rather than a boolean because a single "delivered"
row has to be written on one side of the send, and both sides lose: before it
silently drops the question, after it duplicates forever.

**Alternatives.** All four rejected in DECISION.md were re-checked against the
code rather than re-argued. Two decisions were made here:

- `confirm_delivery` RAISES on a delivery nothing claimed, and is a silent
  no-op on one already confirmed. The first is the same class of error as
  `append_event`'s unknown conversation - with no FOREIGN KEYs, a quiet no-op
  would read as a completed delivery. The second keeps the FIRST confirmation's
  timestamp, which is when the send actually returned.
- No `DeliveryRecord`. Step 6 left it conditional on the shape needing one, and
  it does not: `claim_delivery` returns `bool`, `confirm_delivery` returns
  `None`, and `pending_events` returns `EventRecord`. Nothing outside the
  package reads a delivery row, so a record would be a type with no reader.

**Difficulties.** One, and it was a guard rather than a defect:
`test_migration_creates_the_delivery_table` took `tests/test_db_migrations.py`
to 917 lines against the 900-line test cap, whose allowlist is a ratchet that
may not be grown. Split by SUBJECT rather than by size - `test_db_migrations.py`
keeps the runner (reaching head, the connection the DDL runs on, the backup,
where the scripts ship from) and a new `tests/test_db_schema.py` takes what the
revisions leave behind (the autogenerate diff, the table list, the three
constraint proofs). The `fresh` fixture went to `tests/conftest.py` so the two
share it rather than duplicating it. `scufris/README.md`'s maintainer revision
loop names both files now.

**Evidence.** `nix flake check` - all 6 checks passed, including `filesize`,
`records` and the full `pytest`. Red before the work: `pytest
packages/chat/tests -q -k delivery` exited 5, no tests collected. Green after:
11 tests in `packages/chat/tests`, 25 across the two migration modules,
`examples/chat_conversation.py` exits 0 and prints both channels sending events
1 and 2 and then replaying and sending nothing. Revision `53aaa107ce2d` was
autogenerated against the models, not hand-written, and
`test_schema_has_no_pending_autogenerate_diff` is what holds it to them.

**Reflection.** The plan's two arg orders - `claim_delivery(conn, channel,
conversation_id, event_seq)` against `pending_events(conn, conversation_id,
channel)` - were kept as planned: each follows its own neighbour, the key's
column order and `read_transcript`'s. It reads as an inconsistency at a glance
and is worth revisiting when Lane 2 has a real caller to judge it against.
The delivery table has no test for two channels claiming CONCURRENTLY, where
`test_chat_events.py` has one for `event_seq`; the claim is held by the same
`BEGIN IMMEDIATE` that test already exercises, and the primary key would turn a
double claim into a failed INSERT rather than a duplicate, so the added
coverage would be of SQLite rather than of this store.

### Close-out, round 1

Two passages above are SUPERSEDED by the review round rather than edited away,
since these records are history: the `confirm_delivery` bullet under
Alternatives, and the Reflection's defence of the two argument orders.

**What and why.** R1.1 was a real blocker and the rest followed from it.
`claim_delivery` answered `False` for any existing row, so an abandoned
`claimed` row - the exact state the two-state design exists to make
recoverable - was handed back by `pending_events` and then skipped by the
claim, forever. The table's central promise was inverted: the branch shipped
"silently loses the question", which is the failure DECISION.md section 2
rejects by name. It now answers `True` for a `claimed` row too, and `False`
only for `confirmed`; DECISION.md section 6 records that contract.

The reason it shipped green is R1.2: the test named for this case confirmed
directly instead of re-running the claim path a restart runs, so it proved the
row was pending and never that anything sent it. Rewritten to drive the loop
the README and the example document, it fails `assert [] == [1]` against the
old code - verified by restoring the defect, not by inspection.

**Alternatives, revised.** `confirm_delivery` is no longer a silent no-op on an
already-confirmed delivery; it raises whenever no `claimed` row matches (R1.6).
The branch had no caller and no test, and deleting it was cheaper than
inventing a requirement for it. Its "keeps the first timestamp" rationale went
with it. The `sqlite_insert(...).on_conflict_do_nothing()` form (R1.13) makes
the claim's answer true by construction instead of resting on `BEGIN
IMMEDIATE`, following `scufris/scheduler.py:133`.

**Reflection, revised.** The two argument orders were defended above as each
following its own neighbour. That was wrong to keep: R1.11 is right that two
adjacent functions on one public surface with swapped ids is a foot-gun, and
the moment to align them was while no caller existed. All three now take
`(conn, conversation_id, channel, ...)`.

The deeper lesson is R1.2's, and it is not about this table. The test was
written from the DoD sentence rather than from the code path a restart takes,
so it asserted the state the design produces and skipped the transition that
produces it. A test that drives the documented caller loop would have caught
the blocker at the moment it was written; one that reaches past the loop to the
function under it can only confirm what the author already believed. The
README and the example documented the broken pass too, which is why the defect
was in the contract and not only in the code.

**Evidence.** `nix flake check` - all 6 checks passed ("running 6 flake
checks"). `pytest packages/chat/tests tests/test_db_migrations.py
tests/test_db_schema.py tests/test_examples.py -q` is green at 46, one more
than round 1's 45: `test_migrated_delivery_check_lists_exactly_the_declared_states`
(R1.3). `examples/chat_conversation.py` exits 0 and still prints both channels
sending events 1 and 2 and then replaying and sending nothing.

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 4.
- `tasks/20260804-115319/DECISION.md` settles what NOTES.md left open: the
  three-column key, the two states, the honest guarantee, no channel registry,
  and one channel-facing read.
- This replaces `TelegramApprovals._announced`, an in-memory `OrderedDict` that
  dies on restart (`scufris/telegram/approvals.py:79`). Note it is a HANDLE map
  as much as a dedupe - `action_id -> [(chat_id, message_id)]`, read by
  `announce_decision` to edit the card. This task replaces its idempotency half
  only; the durable home for the message handles is Lane 2's, which is another
  reason this must land first.
- The Lane 2 host approval decoupling depends on this table existing; sequence
  them in that order.
- Deferred with reason: whether a long-offline channel replays every missed
  event or only the unresolved ones. DECISION.md section 4 keeps it open by
  making it a predicate over `pending_events`' result rather than a shape
  decision - it is Lane 8's, and it is why the epic's Lane 8 exists.
- Deferred with reason: no lease or timeout on a `claimed` row. Nothing in this
  package has a clock, and a reaper with no caller is a mode with no
  requirement.
- Lane 1 of `tasks/20260801-154211/TASK.md`. The epic's Lane 1 example also
  wants a rich transcript and a backend switch; both are out of scope here - the
  backend cache does not exist yet, and this task's example section is the
  replay proof only.
- `packages/chat` depends on `scufris_core` alone, and
  `tests/test_package_boundaries.py` is what keeps that true. Nothing in these
  steps adds a dependency.
