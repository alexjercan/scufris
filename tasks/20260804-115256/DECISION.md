# Decision: event granularity, actor kinds, retention, and who opens the transaction

- DATE: 20260804-121112
- STATUS: ACCEPTED
- TASK: 20260804-115256
- TAGS: v0.2.0, chat, schema, architecture

## Context

`tasks/20260729-220835/DECISION.md` ratified the actor-aware conversation and
listed per-turn event granularity and retention policy under "Not addressed
here". Both are load-bearing for THIS task: they are the shape of the `event`
table, and the table cannot be written without settling them. Three smaller
forks are settled here for the same reason - which actor kinds exist, whether
`conversation` carries a `backend`, and who opens the transaction an event is
written in.

Constraints that bound every choice below: `scufris_core.Base` is the one
metadata object and `Database.transaction()` is the one unit of work
(`packages/core/src/scufris_core/engine.py`); the begin is `BEGIN IMMEDIATE`, so
a read-modify-write inside one transaction cannot interleave with another
writer; no FOREIGN KEYs, per `scufris/db/models.py`'s recorded reason; and
`packages/chat` may import `scufris_core` and nothing else.

## Decision

### 1. One event per meaningful thing said, not one per turn

An `event` row is one attributable utterance: an operator message, an agent
report, a system notice. A single turn therefore produces N rows, and the
conversation is read as an ordered transcript rather than reconstructed from
turn blobs.

`tasks/20260729-220835/mockup.html` draws tool calls and agent reports as
separately attributable things, and section 3 of that record makes "who said
this" a typed query that decides whether an event may satisfy a stop gate. A
turn-grained row cannot answer that question for anything inside the turn: the
operator's message and the agent's report would share one actor column. The
finer grain is the only one where the stop gate's refusal is a property of the
data.

`kind` is a plain string column, not an enum. Nothing in this task branches on
it; the enum lands with the first caller that does (`20260804-115320`,
`20260804-115321`).

### 2. Four actor kinds: `operator`, `orchestrator`, `agent:<id>`, `system`

`Actor` is a frozen value with a kind and, for `agent`, an id. It is parsed at
one boundary and unparseable strings raise there; the row additionally carries a
CHECK constraint on the kind column, so a hand-written INSERT cannot introduce a
fifth kind either.

Four, not the three this task's Story names: `tasks/20260729-220835/DECISION.md`
section 3 is the ratified list and names `orchestrator` separately from
`agent:<id>`. Folding the orchestrator into `agent:<id>` would change the
meaning of a ratified record for no gain - the fourth case costs one enum member
and its rejection would have to be re-litigated by the lane that adds the
coordinator.

The channel an `operator` event arrived on is NOT carried yet. It has no reader
until the delivery table exists (Lane 3), and a column no code reads is a claim
that cannot be kept honest.

### 3. `conversation` carries no `backend` column

The conversation is meant to OUTLIVE any backend - that is the point of section
1 of the ratified record, where the provider session is a cache keyed by
`(conversation, backend, policy version)`. Putting `backend` on the conversation
would give the cache's key a second home on the record it is a cache OF, and the
first backend switch would have to decide which one is true. The session cache
in `20260804-115320` owns it.

### 4. The writer takes an open `Connection`; it never opens one

`append_event(conn, ...)` is the only writer, and it takes the caller's open
connection. It cannot open a transaction of its own, so the invariant from
section 4 of the ratified record - the state change and its event commit
together - is structural rather than a rule callers are asked to keep.
`event_seq` is assigned inside that connection as
`COALESCE(MAX(event_seq), 0) + 1` scoped to the conversation, the pattern
`HostActionRow.seq` already uses (`packages/hostctl/.../actions.py:234`), and
`BEGIN IMMEDIATE` is what makes it safe under concurrent writers.

### 5. v0.2.0 deletes no events - a choice, not an oversight

There is no retention window, no compaction and no operator-visible history
limit in this release. The `event` table grows without bound for as long as the
database lives.

This is recorded so a large table later reads as a known consequence. The
release has one operator on one host and a conversation log of semantic events
(not transcripts); the volume that would justify a policy is not reachable
inside v0.2.0, and a retention rule invented before anyone has read a month of
real events would be a guess with a migration attached.

## Alternatives considered

- **One event per turn.** Cheaper to render - one row, no transcript query per
  view - and it is what a chat UI naively wants. Rejected: it cannot attribute
  anything inside the turn, so an agent report and the operator's message that
  followed it share one actor, and the stop gate's "only an `operator` event may
  authorize" becomes a convention the renderer follows. That is the exact defect
  `tasks/20260729-220835/SPIKE.md` located in the shipping code.
- **Actor as a plain string compared at the call site.** Zero code. Rejected for
  the reason the Story gives: `agent:<id>` is what the stop gate refuses, and a
  string comparison spread over call sites is unfakeable nowhere.
- **A `retention_days` column or a reaper now.** Rejected as speculative: no
  requirement names a window, and `HostActionStore._reap` exists because the
  helper's queue has a bound this table does not have.
- **Defer granularity to the first consumer.** Rejected: the table ships in this
  task, and a deferred decision here is a schema migration in the next one.

## Consequences

**Gained.** "Who said this" and "may this authorize a gate" are typed queries
against a CHECK-constrained column. A rolled-back unit of work consumes no
sequence number, because the number is read and written inside it. Tool calls
and reports are separately attributable, which is what the mockup already draws.

**Paid.** A render reads N rows per turn instead of one, and the transcript
query becomes the thing Lane 6 has to keep bounded. The `event` table has no
upper bound in this release. The `orchestrator` kind has no writer until the
coordinator lands, so it is a case the enum carries before anything produces it.

**Reversal.** Granularity is the only expensive one to reverse: coarsening it
later means collapsing rows, which loses attribution. Retention is additive - a
policy is a new task, not a change to this schema. The `backend` column can be
added to `conversation` if the cache turns out to need a second home.
