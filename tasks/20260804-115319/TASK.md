# Deliver chat events to every channel exactly once

- PRIORITY: 99
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256

## Story

As the operator, I want one conversation event to reach every channel exactly
once, so that a restart mid-delivery does not duplicate a card and a channel
that was offline does not silently miss the question it was supposed to ask me.

## Steps

- [ ] Add failing tests first, then the `delivery` table.
- [ ] Key it `PRIMARY KEY (channel, idempotency_key)` with
      `idempotency_key = (conversation_id, event_seq)`. The key is DERIVED from
      the event, not minted per attempt: that is what makes a retry after a
      crash collide rather than duplicate.
- [ ] Commit the state change and its event in ONE transaction. A delivery row
      that exists without its event, or an event whose delivery was lost, are
      both reachable if this is two transactions.
- [ ] Make replay a no-op at the storage layer rather than at each channel.
      Every channel gets the same guarantee without implementing it twice, and
      a new channel cannot forget.
- [ ] Grow `examples/chat_conversation.py` to show a replayed delivery
      changing nothing, with an assertion behind it.

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

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 4.
- This replaces `TelegramApprovals._announced`, an in-memory `OrderedDict` that
  dies on restart. The Lane 2 host approval decoupling depends on this table
  existing; sequence them in that order.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
