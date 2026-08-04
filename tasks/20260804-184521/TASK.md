# Retire the two open round-1 minors on the delivery contract

- PRIORITY: 35
- TAGS: feature, v0.2.0, lane1, chat
- ACTIVITY: WORKING
- GATES: PLAN
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
      because no non-immediate engine exists. The qualification goes in the
      FIRST paragraph, beside the promise it qualifies - the last paragraph
      already names `scufris_core.engine` and a reader who stops at paragraph
      one never reaches it.
- [ ] 2. R1.2. `tasks/20260804-115319/DECISION.md:127`. Section 6 still reads
      "no correct caller reaches that, because every one gates its send on a
      `True` claim, and a `True` claim always leaves the row `claimed`", which
      `test_two_overlapping_passes_over_one_channel_both_complete` falsifies.
      Append one dated line under that sentence pointing at
      `tasks/20260804-141639/DECISION.md`. Append, do not rewrite: task records
      are append-only history, which is the constraint that left this open.
      The note goes DIRECTLY under that sentence, not at the end of the file:
      the cold reader who lands on section 6 is the one being corrected.
      `tasks/20260801-100405/DECISION.md:25` and `:135` carry the blockquote
      form this repository already uses for an in-place correction.
- [ ] 3. Close both findings where they were raised.
      `tasks/20260804-141639/REVIEW.md` R1.1 and R1.2 are still `- [ ]` with an
      empty `- Response:`, so that record reports two open findings after they
      are retired - the same stale-surface failure Steps 1 and 2 exist to fix.
      Tick both and fill each `- Response:` naming this task and the surface it
      changed, matching the `Response: fixed ...` form used across
      `tasks/*/REVIEW.md`. Append to the response lines only; leave the finding
      text and the round's APPROVE verdict untouched.

## Definition of Done

- `claim_delivery`'s first docstring paragraph names the engine assumption its
  completed-delivery no-op rests on
  (cmd: `grep -A8 'Take responsibility for sending one event' packages/chat/src/scufris_chat/store.py | grep -q 'scufris_core.engine'`)
- That qualification and the last paragraph's `IntegrityError` caveat describe
  one contract rather than two disagreeing ones
  (manual: read the first and last docstring paragraphs side by side; the
  `False`-only-for-`confirmed` promise and the deferred-begin caveat agree)
- `tasks/20260804-115319/DECISION.md` section 6's falsified closing clause
  carries its dated pointer to the record that supersedes it
  (cmd: `grep -A6 "no correct caller" tasks/20260804-115319/DECISION.md | grep -q "tasks/20260804-141639/DECISION.md"`)
- `tasks/20260804-141639/REVIEW.md` reports no open round-1 finding
  (cmd: `grep -q '^- \[x\] R1\.1' tasks/20260804-141639/REVIEW.md && grep -q '^- \[x\] R1\.2' tasks/20260804-141639/REVIEW.md`)
