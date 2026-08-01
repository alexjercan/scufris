# Retro: Land the Alembic migration runner and the projects schema

- TASK: 20260801-120404
- BRANCH: fix/alembic-migration-runner
- REVIEW ROUNDS: 2

## What went well

- The in-package Alembic environment was checked from OUTSIDE the repo against
  the built store path before the branch was offered for review, which is the
  only place the `only-include = ["scufris"]` claim can be tested.
- Round 2 moved sabotage into the test-writing step, as round 1's process signal
  asked. Five fixes, five falsifications, and two bad proofs were caught while
  they were being written rather than at verification.
- Reasoned pushback held up. R1.2's suggested fix (a plain `engine.connect()`
  pre-check) was implemented, failed, and the failure became the argument: the
  round-2 reviewer reproduced it independently at 5.006s versus 0.000s.
- Both rounds found real defects no check could have: the fresh-database WAL
  race, the world-readable backup window, and the newer-revision backup were all
  invisible to a green suite.

## What went wrong

- Two of three load-bearing proofs passed under their own sabotage on first
  write, for the second task running on this lane. Both asserted the good
  OUTCOME (a table exists; no `.bak`) rather than the mechanism producing it.
- `PRAGMA journal_mode=WAL` not invoking the busy handler was not known when
  `engine.py` was written, and the comment there asserted the opposite. The
  decision seemed sound: `busy_timeout` was set FIRST precisely so everything
  after it waits, and SQLite documents no exception in that sentence. It took
  four concurrent OS processes to see, and it only bites on the one-time
  delete->WAL conversion of a FRESH database - the case a single-process test
  suite never reaches.
- Fixing R1.2 by leaving SQLAlchemy silently changed the exception TYPE on the
  startup path, contradicting a corruption contract stated in two docs. Round 2
  caught it. A fix that moves a read off a documented boundary should have
  prompted re-reading what that boundary promises.
- A repo-wide `ruff format .` swept five unrelated test modules into the working
  tree. They were split into their own commit ahead of the fix commit, but only
  because the commit's file list was read afterwards.
- The round-2 record first restated the round-1 finding ids as `- [x] R1.x`
  checkboxes under `## Round 2`; `tatr check --ledger` reads those as findings of
  that round and failed the records check with nine errors.

## What to improve next time

- Write the sabotage in the same edit as the assertion, not in a later
  verification pass. A proof is not finished until it has failed once.
- When a fix changes WHICH layer performs an operation, grep the docs for what
  that layer promises. The word to grep here was `DatabaseError`.
- `git status --short` before the commit message is written, not after.
- Verifying an earlier round's findings belongs in prose under the new round;
  only the new round's own findings take checkbox ids.

## Action items

- None requiring a task. The one disclosed limitation - the
  `upgrade_to_head -> backup_database` wiring has no end-to-end exercise until a
  second revision exists - is recorded in TASK.md and belongs to the next
  revision, which is the following task in this lane.

## Diagnose

- **Breadth.** The diff is a schema, a runner, an Alembic environment and their
  proofs. It is one inherently indivisible feature: splitting the runner from
  the first revision would have landed a runner with nothing to run, and the
  drift proof needs both halves. Not a missed split.
- **Churn.** The rework came from things the plan could not have known - a
  SQLite pragma exception, a umask window, an unknown-revision path - not from a
  design the plan encoded wrongly. The one plan-level miss is Step 5, whose two
  clauses contradicted each other (set a URL / do not let env.py open its own
  engine); `plan`'s from-scratch challenge on that Step would have caught it, and
  the Step is now amended rather than carried on intent.
- **Context.** One compaction occurred, during the round-1 fix pass, with a
  working set of the runner plus its tests plus REVIEW.md. Nothing was lost - the
  pass resumed from the records - but the round-1 fix pass is the point to hand
  off next time rather than the verification after it.
