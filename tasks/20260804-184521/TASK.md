# Retire the two open round-1 minors on the delivery contract

- PRIORITY: 35
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-141639

## Story

As Lane 2's channel author, I want both delivery contract surfaces to be true
before I write against them, so that I do not inherit a promise the code only
keeps under one engine or a decision record whose closing sentence the tests
falsify.

Both findings are round-1 MINORs on `tasks/20260804-141639/REVIEW.md`, raised
independently by two out-of-context review lanes and left open at APPROVE.

## Steps

- [ ] 1. R1.1. `packages/chat/src/scufris_chat/store.py:233`, `claim_delivery`'s
      first docstring paragraph. It promises "``False`` only for a ``confirmed``
      row, which is what makes a replay of a completed delivery a no-op at the
      STORAGE layer" without qualification. Since `20260804-141639` deleted the
      conflict loser's re-SELECT, a claimant under a foreign engine's deferred
      begin that read `None`, lost the INSERT race, and whose winner then
      confirmed answers `True` for a `confirmed` row. Qualify the promise with
      the immediate-begin assumption `scufris_core.engine` supplies, matching the
      last paragraph's existing qualification of the `IntegrityError` case.
      Prose only; no behavior change, and no test exists or is possible in-tree
      because no non-immediate engine exists.
- [ ] 2. R1.2. `tasks/20260804-115319/DECISION.md:127`. Section 6 still reads
      "no correct caller reaches that, because every one gates its send on a
      `True` claim, and a `True` claim always leaves the row `claimed`", which
      `test_two_overlapping_passes_over_one_channel_both_complete` falsifies.
      Append one dated line under that sentence pointing at
      `tasks/20260804-141639/DECISION.md`. Append, do not rewrite: task records
      are append-only history, which is the constraint that left this open.

## Definition of Done

- `claim_delivery`'s docstring states the engine assumption its
  completed-delivery no-op rests on
  (manual: read the first and last docstring paragraphs side by side; the
  `False`-only-for-`confirmed` promise and the deferred-begin caveat agree)
- No live surface claims a correct caller cannot reach `confirm_delivery`'s
  raise
  (manual: `grep -rn "no correct caller" packages/` returns nothing, and the
  only `tasks/` hit carries its dated round-2 note)
