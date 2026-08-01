# Review: Bump the pinned tatr to v2 and disposition the lessons ledger

- TASK: 20260731-175511
- BRANCH: master (no feature branch; the task carries no code diff)

## Round 1

- REVIEWER: in-context
- VERDICT: APPROVE

No implementation to critique: the record was picked up already satisfied, so
the review is the falsification of that claim rather than a read of a diff.

Each DoD proof was run on `master` at 95757a8, with this record itself at
`STATUS: IN_PROGRESS` so the `unplanned-in-progress` finding the task exists to
kill would fire if it were still live:

- `tatr version` -> `tatr 0.2.0`; `nix develop -c tatr version` -> `tatr 0.2.0`.
  The pinned and local versions agree.
- `tatr check --ledger LESSONS.md` -> no findings, exit 0.
- `nix flake check` -> `all checks passed!` (7 checks, `scufris-records`
  included).

Provenance of the work, so the close is not an unexplained green:

- The pin moved to `github:alexjercan/tatr/v0.2.0` in `flake.nix:33`, with
  `flake.lock` resolving `ref: v0.2.0`. `git log -S'tatr/v0.2.0' -- flake.nix`
  attributes it to 1253dfd.
- All 10 `promotion-awaiting-decision` entries under
  `## Pending promotions` now carry an operator disposition
  (PROMOTE / DEFER / ABSORBED); the PROMOTE calls name 20260731-233221 as the
  task that carries them into repository guards.
- The `bad-disposition` entry reads
  `isolate-state_dir-in-tests-that-assert-config (x3, DEFER 2026-07-31 ...)`,
  so it has both the count and the disposition v0.2.0 requires.

The one thing this task did contribute is the negative result above: the gate
is now green *while* a task is IN_PROGRESS, which is the property the Story
asked for and which no earlier task verified in that state.
