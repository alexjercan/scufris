# Retro: Bump the pinned tatr to v2 and disposition the lessons ledger

- TASK: 20260731-175511
- BRANCH: master (no feature branch; the task carries no code diff)
- REVIEW ROUNDS: 1

## What went well

- The DoD was written as three runnable `cmd:` proofs rather than as prose
  about a version bump. That is the only reason the task could be settled by
  measurement instead of by argument: three commands answered "is this still
  needed" in under a minute, with no reading of the flake or the ledger
  required to reach the verdict.
- The proofs were run with this record at `STATUS: IN_PROGRESS`, which is the
  exact state the Story cares about. Running them between tasks would have
  reproduced the old false green the task was written to eliminate.

## What went wrong

- **The record outlived its premise and nothing noticed.** Between creation and
  pickup, the pin bump rode in with an unrelated task (1253dfd, a frontend
  split) and the ledger dispositions were made under 20260731-233221. Neither
  closed nor annotated this record, so a task sat at priority 92 describing a
  red gate that was already green.
  The decision that seemed sound at the time: 20260731-171420 deliberately left
  the pin alone rather than drag an operator-owned ledger decision into a
  file-size guard, and spun this record out instead. That was right. What was
  missing is the other half - when a later task DOES take the deferred work,
  the spun-out record is the thing that has to be revisited.

## What to improve next time

- Run a task's DoD proofs BEFORE planning it, not after. A proof that is
  already green on the base branch means the task is done, not that the plan is
  easy. The plan skill already requires the proofs to be red on base; treat a
  green one as a close signal rather than as a step to skip.
- When a task lands work that another OPEN record claims, name that record in
  the landing commit or in its Notes. A deferred task is a promise the deferrer
  owns; whoever redeems it early owes the record an update.

## Action items

- 20260731-233221 (`Promote the recurring lessons into repository guards`) is
  still BACKLOG at priority 60 and carries every PROMOTE disposition this
  task's Steps handed to the operator. The ledger records the calls; that task
  is what turns them into guards. Nothing here blocks it.
- Both lessons below are x1 and land in the Build / environment section of
  `LESSONS.md`, not in Pending promotions. Neither is at the 3x promotion bar.

## Lessons

- `a-green-dod-proof-on-base-means-the-task-is-done` (x1): the plan skill's
  rule that every `cmd:` proof must be RED on the base branch is not only a
  proof-quality check - a proof that comes back green is the task reporting
  that it has already been satisfied, by a later task, by an upstream bump, or
  by drift. Run the proofs first and read a green as "close this", not as
  "nothing to verify here". 20260731-175511.
- `deferred-work-taken-early-must-update-the-record-that-owns-it` (x1): when a
  task defers work into a spun-out record, that record becomes the owner of the
  promise. A later task that happens to do the work (a version bump riding in
  with an unrelated split, a ledger swept during a different pass) leaves the
  owning record stale and mis-prioritised. Name the record in the commit or in
  its Notes at the moment the work lands. 20260731-175511.
