# `packages/chat/` - the conversation Scufris owns

`scufris-chat` is two tables and four functions. It holds the conversation as
**Scufris'** record rather than as a view onto whichever provider answered last:
a `conversation` outlives every backend session under it, and an `event` is one
attributable thing said inside it.

It depends on [`scufris_core`](../../../core/src/scufris_core/) and nothing else.
No host, no helper, no agent - the conversation is a schema and a store over it,
and `tests/test_package_boundaries.py` is what keeps that true.

- The shape of the tables, and why: [`tasks/20260804-115256/DECISION.md`](../../../../tasks/20260804-115256/DECISION.md).
- The actor-aware conversation this implements: [`tasks/20260729-220835/DECISION.md`](../../../../tasks/20260729-220835/DECISION.md).
- It running, end to end: [`examples/chat_conversation.py`](../../../../examples/chat_conversation.py).

## 1. The two tables

| Table | One row is | Notes |
|---|---|---|
| `conversation` | one durable thread | an id and when it started, and deliberately nothing else |
| `event` | one attributable utterance | ordered within its conversation by `event_seq` |

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

## 5. The surface

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

`ConversationRow` and `EventRow` are **not** exported. With no foreign keys, an
id that names nothing is reachable at both ends, so the store checks what the
schema will not: `causing_event` raises `LookupError` on a `causation_id` that
resolves to nothing in this conversation rather than returning `None` - "this
started something" and "its cause is missing" mean opposite things, and a cause
in another thread is not this transcript's cause - and `append_event` raises on a
`conversation_id` that is not a conversation's rather than minting a transcript
nothing owns.

## 6. What is not here yet

`delivery` (which channel an operator event arrived on) and `activity` are later
lanes. The channel is deliberately not a column on `event` today - it has no
reader until the delivery table exists, and a column no code reads is a claim
that cannot be kept honest.

There is also **no retention policy**: v0.2.0 deletes no events, and the table
grows for as long as the database lives. That is a recorded choice, not an
oversight - the release has one operator on one host, and a rule invented before
anyone has read a month of real events would be a guess with a migration
attached.
