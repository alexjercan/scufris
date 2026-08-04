# Decision: the delivery table, its two states, and the honest guarantee

- DATE: 20260804-115319
- STATUS: SUPERSEDED by tasks/20260804-141639/DECISION.md
- TASK: 20260804-115319
- TAGS: v0.2.0, lane1, chat, delivery, idempotency

## Context

`tasks/20260729-220835/DECISION.md` section 4 fixes the key
(`PRIMARY KEY (channel, idempotency_key)` with
`idempotency_key = (conversation_id, event_seq)`) and the transaction rule (the
state change and its event commit together). It fixes nothing else, and
`tasks/20260804-115319/NOTES.md` left two questions open on purpose: what a
channel does with a delivery it has CLAIMED but not yet SENT, and whether a
channel that was offline replays everything it missed. This record settles the
first, and deliberately does not settle the second - it picks the shape that
leaves it open.

## Decision

### 1. The key is three columns, not a rendered string

`delivery` is `PRIMARY KEY (channel, conversation_id, event_seq)`. That IS the
ratified key with its `idempotency_key` half stored as its two parts rather than
as `f"{conversation_id}:{event_seq}"`.

A rendered string would need a parser to go the other way, would make
"everything this channel has for conversation C" a `LIKE` rather than a prefix
scan of the primary key, and would let a conversation id containing the
separator collide. The parts cost nothing: SQLite's primary key over three
columns is one index, and it is the index every per-channel read wants.

The key stays DERIVED from the event and is never minted per attempt. That is
the whole mechanism - a retry after a crash recomputes the same three values and
collides, where a per-attempt key would produce a second row and a second card.

### 2. Two states, claimed then confirmed - not a boolean

A row is written `claimed` inside the same transaction as the event, and moved
to `confirmed` after the channel's send returns. The alternative considered and
rejected was one row meaning "delivered", which has to be written either before
the send or after it, and both are wrong:

| Write | Crash mid-send | Result |
|---|---|---|
| before the send | row says delivered, nothing was sent | the operator silently never sees the question |
| after the send | sent, no row | the retry re-sends and a second card appears, forever |

Two states make the crash window READABLE instead of choosing which way to lose:
a `claimed` row with no `confirmed` is exactly "someone was mid-send when we
died", which is the one case a restart must retry.

This is the requirement the Story names in its own words - "a restart mid-
delivery does not duplicate a card and a channel that was offline does not
silently miss the question" - so the second state has a caller in this task and
is not speculative generality.

### 3. Exactly-once on the normal path, at-least-once across a crash mid-send

Stated plainly because the task title says "exactly once" and the table cannot
deliver that alone. A side effect that has happened cannot be un-happened by a
transaction that rolls back, so no storage-layer design makes an external send
exactly-once without the channel's cooperation.

What this table guarantees:

- A replay of an already-`confirmed` event is a no-op, at the storage layer, for
  every channel. This is the common case and it is exact.
- A crash between the send and the `confirm` re-sends once on restart. Duplicate,
  not lost.

That trade is deliberate and is the direction the Story picks: a duplicated card
is noise, a missed approval request is a stop gate that never opens. Collapsing
the duplicate is the CHANNEL's job with the channel's own affordance - Telegram
edits the message it already posted, which is what
`ApprovalSurface._announced` reaches for today (`scufris/telegram/approvals.py:79`)
from memory that dies on restart. Lane 2 gives that map a durable home; this
task gives it the key to hang it on.

### 4. No channel registry: a row exists only where a channel attempted

Nothing declares the set of channels, and no delivery rows are fanned out when
an event is appended. A row appears when a channel claims one.

So "what has this channel not been told" is a left join - events with no
delivery row for that channel - and it answers correctly for a channel that did
not exist when the event was written, which a fan-out at append time cannot do
without knowing every future channel's name.

This is also what keeps NOTES' second open question open. Whether a long-offline
channel replays every missed event or only the unresolved ones becomes a
predicate over that query's result, chosen by the caller in Lane 8, rather than
a shape this task has already committed to.

### 5. One read, not two

The channel-facing read is one function: events this channel should send now -
those with no delivery row, plus those with a `claimed` row that was never
confirmed. Two functions (`undelivered`, `pending`) were considered and rejected:
every caller wants their union, no caller in the epic wants either half alone,
and two names invite a channel to ask one and forget the other, which is the
per-channel forgetting this whole table exists to prevent.

### 6. The claim answers "should I send", not "did I mint this row"

Added in review round 1, where reading it the other way was the branch's
blocker. `claim_delivery` returns `True` both for a row it minted and for an
existing `claimed` row nobody confirmed, and `False` only for `confirmed`.

Those two `True` cases are one instruction to the caller - send this - and
section 2's whole point is that they are the same situation seen at different
times. A claim that reported minting instead would refuse an abandoned row,
which `pending_events` correctly keeps handing back, so the channel would skip
it on every restart and the operator would never see the question. That is
section 2's "silently loses the question" arriving through the read path rather
than through the write, and it makes `False` mean two incompatible things:
"already done, skip" and "someone else died holding this, skip".

The consequence is that the caller needs no way to tell the two apart, which is
why no delivery record and no state accessor are exported. A re-claim restamps
`claimed_at` so the column means "when the live attempt started" rather than
"when the first, dead one did" - the timestamp a lease would read, if the lane
that gains a clock ever adds one.

`confirm_delivery` is the mirror: it raises unless a `claimed` row matches, and
no correct caller reaches that, because every one gates its send on a `True`
claim, and a `True` claim always leaves the row `claimed`.

## Alternatives considered

- **A single rendered `idempotency_key` string.** Rejected in section 1: needs a
  parser to go back, turns a primary-key prefix scan into a `LIKE`, and lets a
  conversation id containing the separator collide.
- **One boolean "delivered" row.** Rejected in section 2 - it has to be written
  either before the send (silently loses the question) or after it (duplicates
  forever), and the two-state row is what makes the crash window readable rather
  than making that choice.
- **Fanning out a delivery row per channel when the event is appended.**
  Rejected in section 4: it needs a declared channel set, and it answers wrongly
  for a channel that did not exist when the event was written.
- **Separate `undelivered` and `pending` reads.** Rejected in section 5: no
  caller wants either half alone, and two names invite a channel to ask one and
  forget the other.
- **Each channel dedupes for itself.** This is what the shipping code does
  (`scufris/telegram/approvals.py:79`) and it is the counter-example the table
  exists to replace: it dies on restart, and a channel added later reimplements
  it or forgets.

## Consequences

**Gained.** Idempotency is a property of the storage layer, so a channel added
later cannot forget it. The crash-mid-send window becomes a queryable state
rather than an unrecoverable guess. Lane 2's approval decoupling gets the real
idempotency key it depends on.

**Paid.** Every delivery is two writes, and the second one is on the channel's
path rather than the event writer's. A channel that claims and then dies without
ever restarting leaves a `claimed` row that no one reconciles; there is no
timeout or lease, because nothing in this task has a clock to hang one on.

**Not addressed here.** Retention (`delivery` grows with `event`, under the same
non-decision), the replay policy for a long-offline channel (Lane 8), and where
a channel's own message handles live (Lane 2).

**Reversal.** Dropping the table and having each channel dedupe in memory is
exactly today's code, and is the counter-example this replaces.
