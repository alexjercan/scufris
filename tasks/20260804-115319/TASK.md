# Deliver chat events to every channel exactly once

- PRIORITY: 99
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256

## Story

As the operator, I want one conversation event to reach every channel exactly
once, so that a restart mid-delivery does not duplicate a card and a channel
that was offline does not silently miss the question it was supposed to ask me.

## Steps

- [ ] Write `packages/chat/tests/test_chat_delivery.py` FIRST, red, over the
      `database` fixture pattern `test_chat_events.py` already uses (file-backed
      `open_database`, tables from `Base.metadata`, `OWNED_TABLES` grown to
      `("conversation", "event", "delivery")`). Six tests, named in Definition of
      Done. `pytest packages/chat/tests -q -k delivery` currently exits 5, no
      tests collected - that is the red this step turns green.
- [ ] Add `DeliveryRow` to `packages/chat/src/scufris_chat/models.py`:
      `channel`, `conversation_id`, `event_seq` as a composite
      `PrimaryKeyConstraint`, plus `state`, `claimed_at`, `confirmed_at`. No
      FOREIGN KEYs, for the reason that module's docstring already records.
      The key is DERIVED from the event, not minted per attempt: that is what
      makes a retry after a crash collide rather than duplicate. Three columns
      rather than a rendered `idempotency_key` string - DECISION.md section 1.
- [ ] Constrain `state` with a CHECK over the two values, rendered from the
      declaring enum the way `_ACTOR_KIND_CHECK` is rendered from
      `ACTOR_KIND_VALUES`, so a third state cannot land with the constraint
      still naming two. Pair it with a CHECK that `confirmed_at IS NOT NULL`
      exactly when `state = 'confirmed'` - the same both-halves reasoning as
      `_ACTOR_AGENT_ID_CHECK`, and for the same reason: the store rebuilds every
      row into a record.
- [ ] Add three functions to `packages/chat/src/scufris_chat/store.py`, each
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
- [ ] Refuse a delivery whose event does not exist, with a `LookupError` naming
      the channel and the key. There are no FOREIGN KEYs here, so this is the
      store's check to make - the same one `append_event` makes for
      `conversation_id` and `causing_event` makes for `causation_id`, and made
      inside the caller's unit of work so an event appended in it is visible.
- [ ] Export the three functions and the `DeliveryRecord` (if the shape needs
      one) from `packages/chat/src/scufris_chat/__init__.py`, and from
      `__all__`. No row class is exported; `DeliveryRow` stays private, as
      `EventRow` is.
- [ ] Generate the Alembic revision with `down_revision = "18c9104709b8"` and
      confirm `test_schema_has_no_pending_autogenerate_diff` is green - that
      test is what proves the revision matches the models rather than a
      hand-edit that drifted.
- [ ] Grow `tests/test_db_migrations.py`: add `"delivery"` to
      `test_declared_tables_are_the_only_ones` (and correct its docstring, which
      currently states `delivery` is deliberately absent), and add
      `test_migration_creates_the_delivery_table` asserting the composite
      primary key and both CHECKs by INSERTing against them, as
      `test_migration_creates_the_chat_tables` does - a constraint SQLite parsed
      but does not enforce would still appear in `sqlite_master`.
- [ ] Grow `examples/chat_conversation.py` with a delivery section: two channels
      each claim and confirm the same event, then the SAME delivery is replayed
      and changes nothing, with an assertion behind it and a non-zero exit if it
      does not hold. Keep the script offline and `scufris`-free; it is gated by
      `tests/test_examples.py`.
- [ ] Add a section 5 to `packages/chat/src/scufris_chat/README.md` for the
      table, the two states, and the guarantee stated honestly - exactly-once on
      the normal path, at-least-once across a crash mid-send. Link DECISION.md.
- [ ] Run `pytest packages/chat/tests tests/test_db_migrations.py
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
  (test: `tests/test_db_migrations.py::test_migration_creates_the_delivery_table`).
- `delivery` is the only table the schema gained
  (test: `tests/test_db_migrations.py::test_declared_tables_are_the_only_ones`).
- `examples/chat_conversation.py` exits 0 and prints a replayed delivery
  changing nothing (test: `tests/test_examples.py`).

  cmd: `pytest packages/chat/tests tests/test_db_migrations.py tests/test_examples.py -q`
  (red on base: `pytest packages/chat/tests -q -k delivery` exits 5, no tests
  collected)

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
