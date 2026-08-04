# Record the chat conversation and event tables with typed actors

- PRIORITY: 100
- TAGS: feature,v0.2.0,lane1,chat
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211

## Story

As the operator, I want every meaningful thing said in a conversation recorded
as a semantic event with a typed author, so that Scufris owns the conversation
rather than borrowing the provider's, and so that "who said this" is a fact the
database enforces instead of a convention the renderer follows.

This is the first task of Lane 1 and the first table in `packages/chat`. It is
built ALONGSIDE the old stack: `conversation`, `event`, `delivery` and
`activity` collide with no existing `__tablename__`, so nothing is deleted here.

## Steps

- [ ] Settle PER-TURN EVENT GRANULARITY and record it in `DECISION.md`. One
      event per turn, or one per meaningful thing said? This cannot be deferred:
      it IS the shape of the `event` table. The mockup shows tool calls and
      agent reports as separately attributable, which argues against one row per
      turn, but the cost of the finer grain is a transcript query per render.
      Decide, write down which alternative was rejected, and why.
- [ ] Record the RETENTION NON-DECISION in the same `DECISION.md`. v0.2.0
      deletes no events. The `event` table grows without bound. That is a
      CHOICE - the release has no operator-visible history limit and no
      compaction - and it is written down so it reads as a decision rather than
      an oversight when the table is large.
- [ ] Add failing tests first, then the `conversation` and `event` tables
      against `scufris_core.Base`.
- [ ] Type the actor. `operator`, `agent:<id>`, `system` are distinct cases, not
      strings compared at the call site. An `agent:<id>` actor is what the stop
      gate refuses later, so the type is the mechanism, not documentation.
- [ ] Assign `event_seq` monotonically per conversation INSIDE the writing
      transaction. It is not a timestamp, not a UUID and not assigned by the
      caller; two concurrent writers must not be able to observe the same seq.
- [ ] Carry `correlation_id` and `causation_id` on every event, so a report can
      name the request that caused it.
- [ ] Add `examples/chat_conversation.py` in MINIMAL form and register it in
      `tests/test_examples.py` under `OFFLINE` and `EXAMPLES_BY_MEMBER`. This is
      not optional polish: the moment `packages/chat` exists as a workspace
      member, `test_every_member_has_an_example` goes red without it. Lane 1's
      deliverable task grows this same file into the lane demo.

## Definition of Done

- `event_seq` is per-conversation, gap-free and strictly increasing under
  concurrent writers
  (test: `test_event_seq_is_monotonic_under_concurrent_writers`).
- The seq is assigned inside the same transaction that inserts the event, so a
  rolled-back write consumes no number
  (test: `test_rolled_back_event_consumes_no_seq`).
- An actor is a typed value; an unknown actor string cannot be persisted
  (test: `test_actor_must_be_a_known_kind`).
- A caused event resolves to the event that caused it
  (test: `test_causation_resolves_to_the_causing_event`).
- The chat example runs offline and is claimed by the member gate
  (cmd: `python -m pytest tests/test_examples.py`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` sections 1 and 4. Section 4 is
  explicit that the state change and its event commit in ONE transaction.
- `packages/chat` depends on `core` only. It must not import `flow`, `agents` or
  any sibling's `models` or `repo` module; `tests/test_package_boundaries.py`
  enforces this.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
