# Retro: Fix the record lint that reddens nix flake check

- TASK: 20260803-022018
- BRANCH: master
- REVIEW ROUNDS: 1

## What went well

- The Definition of Done was one falsifiable command, so establishing that the
  work was already done cost one `tatr check` rather than an investigation.

## What went wrong

- The task outlived its own fix by a day. The defect was in another task's
  record, that task fixed it while finishing its work, and nothing connected the
  two - so a closed bug stayed on the schedule and got carried through a sprint
  re-cut as if it were real work.

## What to improve next time

- When a bug is filed against ANOTHER task's artifact, the fix will usually
  arrive from that task. Note the dependency in the record so the next reader
  checks it before picking the work up.
- Re-run the Definition of Done before starting, not after. For a
  single-command DoD this is free and would have retired this record
  immediately.

## Action items

- [ ] None. The lint is clean and the guard that catches it - `tatr check`
      inside `nix flake check` - already works as intended.
