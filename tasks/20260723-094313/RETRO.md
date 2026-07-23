# Retro: BC4 auto-wake bridge

- TASK: 20260723-094313
- BRANCH: feat/wake-bridge
- REVIEW ROUNDS: 1 (APPROVE, clean)

## What went well

- Read the supervisor's exact completion-callback ordering BEFORE designing. The
  two scariest requirements - never hold ORCHESTRATOR_ID at launch (self-deadlock),
  and drain-when-idle - turned out to be by-construction guarantees once I
  confirmed `_execute`'s `finally` calls `release()` and sets `run.state=DONE`
  before `on_complete`. The reviewer independently reached the same "safe by
  construction" conclusion.
- Split the bridge into a PURE class with injected collaborators (is-busy, launch).
  That made the tricky logic (defer/batch/409-absorb) deterministically
  unit-testable, and left just the wiring for one async integration test - which
  was sabotage-verified (stub the on_run_complete call -> the ON test fails).
- Firing the wake from the completion callback (not at request_input) is what makes
  deferred wakes self-drain: the orchestrator's OWN turn ending is a completion, so
  it drains the queue exactly when it becomes free. Reusing BC2's run-id-keyed
  WAITING preservation meant the outcome the bridge reads is already correct.

## What went wrong

- The Write tool appended a stray `</content>` line to THREE new files this cycle
  (wake.py, test_wake.py, REVIEW.md) - the first two failed collection with a
  SyntaxError before I caught them. Cost a round-trip each time.

## What to improve next time

- After Write-ing a NEW file, glance at its tail (or grep for a stray closing
  `</content>`/tag) before running - same reflex as the non-ASCII sweep. A stray
  trailing tag SyntaxErrors a .py at collection.

## Action items

- [x] Ledger: `grep-new-files-for-a-stray-write-tag` (x1) - check a freshly
  Written file's tail for a leaked closing tag before running it.
- BC5 (end-to-end example + acceptance test) is the only remaining spike task; it
  will exercise BC1-BC4 together as the stalled-merge scenario.
