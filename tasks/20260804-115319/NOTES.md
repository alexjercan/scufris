# Understanding: idempotent delivery

## What changes

`packages/chat` gains `delivery`. One event reaching two channels becomes a fact
the storage layer guarantees rather than something each channel implements and
one of them gets wrong.

## Surfaces

- `packages/chat/src/scufris_chat/` - the `delivery` table and the write path.
- Its own Alembic revision, after `20260804-115256`'s.
- `examples/chat_conversation.py` - grows a replay section.

## Data and interfaces

`PRIMARY KEY (channel, idempotency_key)` with
`idempotency_key = (conversation_id, event_seq)`.

The key is DERIVED from the event, never minted per attempt. That is the whole
mechanism: a retry after a crash recomputes the same key and collides, where a
per-attempt key would produce a second row and a second card.

The state change and its event commit in ONE transaction
(`tasks/20260729-220835/DECISION.md` section 4). A delivery row without its
event, and an event whose delivery was lost, are both reachable if this is two
transactions.

## Sketches

```
  event_seq=7 written
        |
        +--> delivery (channel=web,      key=(c,7))  INSERT
        +--> delivery (channel=telegram, key=(c,7))  INSERT
                                   |
                        crash + restart, retry
                                   |
        +--> delivery (channel=telegram, key=(c,7))  CONFLICT -> no-op
                                                     no second card
```

## Shape

Idempotency lives at the storage layer, not per channel. Two reasons: every
channel gets the guarantee without implementing it twice, and a channel added
later cannot forget. The alternative - each channel dedupes - is what the
current code does, and the current code is the counter-example.

## Consequences and open questions

- **This replaces `TelegramApprovals._announced`**, an in-memory `OrderedDict`
  that dies on restart. Lane 2's host approval decoupling
  (`packages/hostctl/approvals.py`) consumes this table for its real idempotency
  key, so this task must land before it.
- **Open:** what a channel does with a delivery it has claimed but not yet
  sent. A row that says "delivered" before the send succeeds loses messages; one
  written after loses idempotency across a crash mid-send. Likely a two-state
  row (claimed, then confirmed) rather than a boolean, but that is a design
  decision for the task, not a settled fact.
- **Open:** whether a channel that was offline for a long time replays every
  missed event or only the unresolved ones. Deferring this makes the first
  Telegram reconnect noisy; it is Lane 8's problem but it is decided by the
  shape chosen here.
