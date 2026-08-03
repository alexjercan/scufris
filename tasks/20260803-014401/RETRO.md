# Retro: Make the config-change restart proofs reopen the database and cover the reap bound

- TASK: 20260803-014401
- BRANCH: test/config-change-restart-reap-proofs
- REVIEW ROUNDS: 2

## What went well

- Step 3's escape hatch ("if an assertion other than the new identity one goes
  red, stop and record it") turned the Step 2 failure into a result rather than
  a surprise. It named the suspect wrongly (`action_id`) but the shape of the
  failure right, which was enough.
- Mutation checks carried the "red on the base" proofs without a checkout:
  early-returning `_reap` reddens the bound test, and moving the restarted app
  back inside the first client's `with` block reddens the identity assertion.
- Both diagnoses in DECISION.md 1 were re-derived independently by the reviewer
  and again at compound time, from the code rather than from the record's prose.

## What went wrong

- Step 2 encoded a false premise: that a lifespan shutdown leaves a `building`
  row. The cancellation handler writes `CANCELLED` before re-raising. The plan
  was written from the existing test's docstring, which overclaims, instead of
  from the handler.
- The escape hatch fired but carried no bookkeeping. It said what to stop and
  record; it did not say to file the follow-up the record would promise or to
  amend the DoD clause the stop invalidates. Both were left undone until review
  round 1 (R1.1, MAJOR) - the whole of this task's rework.
- Breadth: not a concern. ~100 lines outside `tasks/`, one production change
  (`abandon_builds` return type), no split available or missed.
- Context: no measured pressure. No compaction warning, checkpoint, handoff, or
  delegation appears in any record. The one context event was mechanical: main's
  TASK.md copy is stale at WORKING while the worktree reads COMPOUNDING, which
  `sprout ls` resolved.

## What to improve next time

- When a plan step asserts a runtime state ("shutdown leaves the row X"), read
  the handler that writes it before planning against it. A test docstring is not
  evidence for the state its test observes.
- A planned escape hatch must name its own bookkeeping: file the follow-up,
  amend the DoD clause it invalidates, annotate the step. Otherwise the finding
  lands in an append-only record no tracker surfaces, and review pays for it.

## Action items

- Follow-up already filed: 20260803-113000, "Prove the startup sweep clears a
  building row orphaned by a crash" (p35, epic 20260729-102145 Lane B).
- Knowledge: new `planning/an-escape-hatch-names-its-bookkeeping`; occurrence
  added to `changes/read-the-implementation-before-reuse` rather than a
  near-duplicate slug, since its body already covers "remembered behavior is not
  a contract" and a test docstring is remembered behavior. `knowledge check`
  exit 0.
