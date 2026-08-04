# Decision: confirm is the mirror of claim, and claim has one write

- DATE: 20260804-182257
- STATUS: ACCEPTED
- TASK: 20260804-141639
- TAGS: v0.2.0, lane1, chat, delivery, idempotency
- Supersedes: tasks/20260804-115319/DECISION.md

## Context

`tasks/20260804-115319/DECISION.md` section 6 closes with the mirror sentence:
`confirm_delivery` "raises unless a `claimed` row matches, and no correct caller
reaches that, because every one gates its send on a `True` claim, and a `True`
claim always leaves the row `claimed`". Round 2 of that task's review falsified
the last clause against the code it describes. Section 6's own earlier half is
why: a claim answers `True` for a `claimed` row it did NOT mint, because an
abandoned row must be handed back or the operator never sees the question. So
two overlapping passes over one channel both get `True`, both send, and the
second `confirm_delivery` raises `LookupError` into an otherwise-correct caller
loop. Reproduced on base at `packages/chat/src/scufris_chat/store.py:313`.

The same review found `claim_delivery`'s conflict-loser branch unreachable -
`scufris_core.engine` opens every transaction `BEGIN IMMEDIATE`
(`packages/core/src/scufris_core/engine.py:266`) - untested, and unable to
answer if it ever were reached: under a deferred begin the loser takes
`OperationalError` on the INSERT rather than `rowcount == 0`, and its re-SELECT
runs on the snapshot that already answered `None`, so `scalar_one()` raises
`NoResultFound`. The docstring's "the answer is true by construction" is not
delivered by that code.

Lane 2 writes the first real channel against these two docstrings. Both are
load-bearing on what that channel has to catch.

## Decision

**`confirm_delivery` accepts an already-`confirmed` row.** When the UPDATE
matches nothing it re-reads through `_delivery_state` and raises `LookupError`
only when there is no row at all. That is the exact mirror of `claim_delivery`:
"should I send" is answered the same way at both ends of the send, and a caller
running overlapping passes needs no way to tell the cases apart - which is the
property section 6 already argued for on the claim side and is why no delivery
record and no state accessor are exported. The raise now fires on exactly one
input, a confirmation of a key that was never claimed, which is the case with no
FOREIGN KEY behind it and the one the refusal was written for.

The `state == CLAIMED` guard stays on the UPDATE, so `confirmed_at` keeps the
FIRST confirmation's time. That differs from `claimed_at`, which a re-claim
restamps, and the reason differs with it: `claimed_at` means "when the live
attempt started", `confirmed_at` means "the send returned", and the earliest
true answer is the one that stays true.

**`claim_delivery` returns `True` straight after the INSERT.** The
`minted.rowcount` test and the re-SELECT go. `on_conflict_do_nothing()` stays -
it costs one clause and turns a conflict under a foreign engine's deferred begin
into a no-op instead of an `IntegrityError` mid-loop - and the docstring claims
only that, instead of claiming a resolution the code cannot perform.

Built from scratch under today's constraints, this is the shape: one write per
claim, one no-branch answer per confirm, and two docstrings a channel author can
take literally.

## Alternatives considered

- **Document the raise instead of tolerating it** (the review's other sanctioned
  option for R2.1). Correcting `confirm_delivery`'s docstring and section 6 to
  say "a second overlapping pass will raise" is honest and is a smaller diff.
  Rejected because it pushes a `try/except LookupError` into every channel from
  Lane 2 onward, to catch a state that is not an error - the delivery HAPPENED,
  and the row already says so. It also makes the raise mean two things at once.
- **Keep the conflict-loser branch and fix it** (the review's other option for
  R2.2) - re-SELECT in a fresh unit of work, or catch `OperationalError`.
  Rejected as a concept with no caller: the only engine in the tree cannot reach
  it, so the fix would be unreachable code with an unreachable test, and YAGNI
  says the later lane that brings a second engine brings the requirement too.
- **Drop `on_conflict_do_nothing()` as well.** Rejected: without it a foreign
  deferred-begin conflict becomes an `IntegrityError` in the middle of a
  channel's loop rather than a no-op, and the clause is one line with a stated
  reason.
- **Do nothing.** All three findings were APPROVEd open as MINOR/NIT, so
  deferring blocks no one today. The cost is that Lane 2's channel gets written
  against a docstring that is false in the one concurrency case the table exists
  to survive, and the wrong `try/except` lands with it.

## Consequences

Easier: the channel loop the README and `examples/chat_conversation.py` document
is now total. Every claim/confirm pair a correct caller can produce completes,
so a channel needs no exception handling around the store, and `claim_delivery`
reads as one read and one write with no branch a reader has to prove unreachable.

Harder: `confirm_delivery` no longer catches a double-confirm inside one pass.
That was never distinguishable anyway - a second confirm of the same row is the
same bytes as the overlapping pass now allowed - but a caller with a genuine
duplicate-confirm bug loses a signal it briefly had. Nothing else replaces it,
and the store deliberately exports no state accessor that would.

This record supersedes `tasks/20260804-115319/DECISION.md` for section 6's
closing mirror sentence ONLY. Everything else in that record - the derived key,
the two states, the honest guarantee, the `True`-for-an-abandoned-row rule - is
unchanged and still the reason this shape is what it is.
