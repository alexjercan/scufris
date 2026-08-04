# `packages/chat/` - the conversation Scufris owns

`scufris-chat` is three tables and seven functions. It holds the conversation as
**Scufris'** record rather than as a view onto whichever provider answered last:
a `conversation` outlives every backend session under it, and an `event` is one
attributable thing said inside it.

It depends on [`scufris_core`](../../../core/src/scufris_core/) and nothing else.
No host, no helper, no agent - the conversation is a schema and a store over it,
and `tests/test_package_boundaries.py` is what keeps that true.

- The shape of the tables, and why: [`tasks/20260804-115256/DECISION.md`](../../../../tasks/20260804-115256/DECISION.md).
- The delivery table, its two states and the guarantee: [`tasks/20260804-115319/DECISION.md`](../../../../tasks/20260804-115319/DECISION.md).
- The actor-aware conversation this implements: [`tasks/20260729-220835/DECISION.md`](../../../../tasks/20260729-220835/DECISION.md).
- It running, end to end: [`examples/chat_conversation.py`](../../../../examples/chat_conversation.py).

## 1. The three tables

| Table | One row is | Notes |
|---|---|---|
| `conversation` | one durable thread | an id and when it started, and deliberately nothing else |
| `event` | one attributable utterance | ordered within its conversation by `event_seq` |
| `delivery` | one channel's attempt at one event | keyed by `(channel, conversation_id, event_seq)`; see section 5 |

`conversation` carries **no `backend` column**. The conversation is meant to
outlive any backend, so the provider session is a cache keyed by
`(conversation, backend, policy version)` and lives with the cache. Putting
`backend` here would give that key a second home on the record it is a cache OF,
and the first backend switch would have to decide which one is true.

`event` is **one row per thing said, not per turn**. A single turn produces
several: the operator's message, the agent's report, a system notice. That is
what makes "who said this" answerable for something *inside* a turn - which is
exactly what a stop gate has to ask, since only an `operator` event may
authorize one. A turn-grained row would give the operator's message and the
agent's report one shared actor column, and the gate's refusal would become a
convention the renderer follows rather than a property of the data.

`kind` is a plain string, not an enum. Nothing here branches on it; the enum
lands with the first caller that does.

There are **no FOREIGN KEYs**, for the reason `scufris/db/models.py` records: a
batch ALTER under `foreign_keys=ON` rebuilds the table the references point at.
`conversation_id` and `causation_id` are plain columns, and what holds them
together is that every write goes through this package inside one transaction.

## 2. `event_seq`, and why it is not a rowid

`event_seq` is the position in the transcript: per conversation, starting at 1,
gap-free and strictly increasing. It is **not** the rowid (SQLite does not
promise to preserve one across a `VACUUM`) and **not** a timestamp (two events
in one transaction share one).

The store assigns it as `COALESCE(MAX(event_seq), 0) + 1`, scoped to the
conversation, **inside the caller's open transaction** - the pattern
`HostActionRow.seq` already uses. Two properties follow, and each has a test:

- **Concurrent writers cannot claim one number.** Every begin is
  `BEGIN IMMEDIATE`, so a read-modify-write cannot interleave with another
  writer's (`test_event_seq_is_monotonic_under_concurrent_writers`).
- **A rolled-back write consumes no number.** The number is read and written in
  the transaction that unwound (`test_rolled_back_event_consumes_no_seq`).

`UniqueConstraint(conversation_id, event_seq)` is the backstop: a store bug
becomes a failed INSERT rather than two events at position 4. Its leading column
is `conversation_id`, so it is also the index every transcript read uses.

## 3. The actor

An author is an `Actor`: a frozen value over four kinds - `operator`,
`orchestrator`, `agent` (which carries an id) and `system`. The wire form it
parses is a bare kind, or `agent:<id>`; the store writes the kind and the id as
two columns, so there is no renderer going the other way.

Two gates, covering different things:

- **`Actor.parse` is the one boundary a string crosses.** An unknown kind raises
  there, an `agent` without an id raises, an id on any of the other three raises,
  and so does a separator with no id after it. No caller downstream has to
  consider a fifth kind.
- **The `event` row carries the same rule as two CHECK constraints**:
  `actor_kind` against the four, and an `actor_agent_id` that is a NON-EMPTY
  string for `agent` and NULL for the other three. The predicate is truthiness
  rather than nullability on purpose: it is the one `Actor.__post_init__` uses,
  and a nullability-only version would accept `''` for an `agent`.
  The parse is code; this is the database, and it is what a
  migration, a repair session or a later store writing the columns directly meets
  instead. Both halves are constrained because `read_transcript` rebuilds every
  row into an `Actor`: one disagreeing row would make the whole conversation
  unreadable, not just itself. The kind list is rendered from the enum, so the
  model and the enum cannot drift, and
  `test_migrated_actor_check_lists_exactly_the_declared_kinds` reads the text off
  a migrated database, which is the half autogenerate does not diff.

`orchestrator` is named separately from `agent` even though nothing writes one
until the coordinator lands. It is the ratified list, and folding it in to save
an enum member would only have to be re-litigated later.

## 4. The connection-passing rule

Every function here takes an **open `sqlalchemy.Connection`** as its first
argument. There is no `Database` on this surface, and nothing in this package
opens a transaction:

```python
with database.transaction() as conn:      # the CALLER owns the unit of work
    conversation = create_conversation(conn)
    append_event(conn, conversation.id, actor=Actor(ActorKind.OPERATOR),
                 kind="message", body="rebuild the dashboard")
```

That is what makes "the state change and the event describing it commit
together" **structural** rather than a rule callers are asked to keep. A caller
writing its own row alongside the event puts both in that block; there is no way
to get one committed without the other having a chance to roll it back. It is
also why the store is functions rather than a class - a class holding a
`Database` would be able to open a second unit of work, which `transaction()`
refuses anyway.

The rules that come with the connection are `scufris_core.engine`'s, unchanged:
a transaction never spans an `await`, loop-thread callers wrap a synchronous
unit of work in `asyncio.to_thread`, and units of work do not nest.

## 5. `delivery`, and what "exactly once" actually means

One row is **one channel's attempt at one event**, and its primary key is
`(channel, conversation_id, event_seq)`. Every part of that key is **derived
from the event**; nothing is minted per attempt. That is the whole mechanism: a
retry after a crash recomputes the same three values and collides with its own
row, where a per-attempt id would produce a second row and a second card.

The two ids are stored as themselves rather than rendered into one
`idempotency_key` string. A string would need a parser to go back, would turn
"everything this channel has for conversation C" into a `LIKE` rather than a
prefix scan of the key, and would let a conversation id containing the separator
collide.

**Two states, not a boolean.** A row is written `claimed` in the same
transaction as the event and moved to `confirmed` once the channel's send
returns. A single "delivered" row would have to be written on one side of the
send, and both sides are wrong:

| Write | Crash mid-send | Result |
|---|---|---|
| before the send | row says delivered, nothing was sent | the operator silently never sees the question |
| after the send | sent, no row | the retry re-sends and a second card appears, forever |

Two states make the crash window **readable** instead of choosing which way to
lose: a `claimed` row with no confirmation is exactly "someone was mid-send when
we died", which is the one case a restart must retry. Both halves are CHECK
constraints - `state` against the two values, rendered from the enum, and
`confirmed_at` against the state - for the same reason the actor columns are.

**The honest guarantee**, since the table cannot deliver "exactly once" alone: a
side effect that has happened cannot be un-happened by a transaction that rolls
back.

- A replay of an already-confirmed event is a **no-op**, for every channel. This
  is the common case and it is exact.
- A crash between the send and the confirm **re-sends once** on restart.
  Duplicate, not lost.

That trade is deliberate: a duplicated card is noise, a missed approval request
is a stop gate that never opens. Collapsing the duplicate is the channel's job
with the channel's own affordance - Telegram edits the message it already
posted.

**A channel's pass** is `pending_events`, then per event `claim_delivery`,
send, `confirm_delivery`. The claim answers `True` for a row it minted *and*
for a `claimed` row nobody ever confirmed, because those are the same
instruction to the caller: send this. It answers `False` only for `confirmed`,
which is what makes the replay exact. A claim that refused every existing row
would strand an abandoned one pending forever and the operator would never see
the question - the failure the second state exists to rule out.

The send goes **between two units of work**, never inside one: claim and
commit, send, then confirm in a second transaction. A transaction spanning the
send would hold the claim unwritten while the card was posted, so a crash would
lose the row that records it and the next pass would post a second card.
`examples/chat_conversation.py` is that loop, runnable.

**Nothing declares the set of channels.** No rows are fanned out when an event is
appended; a row exists only where a channel attempted. So "what has this channel
not been told" is a left join, and it answers correctly for a channel that did
not exist when the event was written - which a fan-out at append time cannot do
without knowing every future channel's name.

`pending_events` is **one read, not two**. It returns events with no delivery row
for the channel *plus* events whose row was claimed and never confirmed. Every
caller wants that union, and two names would invite a channel to ask one and
forget the other, which is the per-channel forgetting this table exists to
prevent. Whether a long-offline channel then sends all of them or only the ones
still unresolved is a predicate over the result, chosen by the caller.

There is **no lease or timeout** on a `claimed` row. A channel that claims and
then dies without ever restarting leaves one that nothing reconciles; nothing in
this package has a clock to hang a reaper on, and a reaper with no caller would
be a mode with no requirement.

## 6. The surface

`scufris_chat` is the whole public surface. A sibling imports the package, never
`scufris_chat.store` or `scufris_chat.models`, and
`test_no_package_imports_a_sibling_private_module` enforces it.

| Name | What it does |
|---|---|
| `Actor`, `ActorKind` | the typed author; `Actor.parse` reads the wire form |
| `ConversationRecord`, `EventRecord` | frozen values; what the store returns |
| `create_conversation(conn)` | mint a thread |
| `append_event(conn, conversation_id, *, actor, kind, body, ...)` | append one utterance, numbered; `LookupError` if the conversation is not there |
| `read_transcript(conn, conversation_id)` | the whole thread, oldest first |
| `causing_event(conn, event)` | the single event this one answers, or `None`; `LookupError` if the id names nothing in this conversation |
| `claim_delivery(conn, conversation_id, channel, event_seq)` | `True` if the caller should send - a row it minted, or a `claimed` one nobody confirmed; `False` only for `confirmed` |
| `confirm_delivery(conn, conversation_id, channel, event_seq)` | called AFTER the send returns; `LookupError` if nothing is sitting in `claimed` |
| `DeliveryState` | the two states the `delivery` CHECK is rendered from |
| `pending_events(conn, conversation_id, channel)` | what this channel should send now, oldest first |

`ConversationRow`, `EventRow` and `DeliveryRow` are **not** exported. With no
foreign keys, an id that names nothing is reachable at both ends, so the store
checks what the
schema will not: `causing_event` raises `LookupError` on a `causation_id` that
resolves to nothing in this conversation rather than returning `None` - "this
started something" and "its cause is missing" mean opposite things, and a cause
in another thread is not this transcript's cause - and `append_event` raises on a
`conversation_id` that is not a conversation's rather than minting a transcript
nothing owns. `claim_delivery` and `confirm_delivery` refuse the same way, for
the same reason: a delivery of something that was never said, and a confirmation
of something that was never claimed, would both read as successful deliveries.

## 7. What is not here yet

`activity` is a later lane. The channel an operator event ARRIVED on is still
deliberately not a column on `event` - `delivery` records where events are sent,
not where they came from, and an inbound column has no reader yet.

There is also **no retention policy**: v0.2.0 deletes no events, `delivery` grows
alongside `event`, and both grow for as long as the database lives. That is a
recorded choice, not an
oversight - the release has one operator on one host, and a rule invented before
anyone has read a month of real events would be a guess with a migration
attached.
