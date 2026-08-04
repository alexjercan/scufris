# Make DoD proofs falsifiable: section-scope document greps and add the plan-time falsifiability question

- PRIORITY: 0
- TAGS: chore,backlog,process,plan
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want every DoD proof to fail when its Step is undone, so
that a green `cmd:`/`test:` line is evidence the Step happened rather than
evidence the pattern exists somewhere in the repository.

## Notes

- Promotion of `dod-proof-must-exercise-the-named-claim` (x3), decided
  2026-08-01. Ledger entry in the Pending promotions section of `LESSONS.md`.
- Occurrences: 20260724-111947 (order claim proved by set membership),
  20260724-132830 (rendering claim proved by an API field), 20260729-102146 (a
  document grep satisfied by a section written for a different Step, so a
  genuinely undone Step was ticked green).
- Three ledger siblings are the same failure in other shapes and should fold
  into one clause rather than stay separate:
  `dod-kfilter-proof-must-select-tests` (a `-k` filter selecting zero tests),
  `scope-absence-greps-to-the-diff-not-the-file` and
  `absence-grep-must-not-be-extension-scoped` (absence greps self-matching
  pre-existing content).
- Candidate guards from the ledger, cheapest first: a plan-skill clause
  requiring the falsifiability question ("could this command pass while the
  Step is undone?") for each proof; a template making document proofs
  section-scoped by default (`rg "^### <heading>"` plus locations appearing
  nowhere else); a checker that flags a `cmd:` proof whose pattern already
  matches at plan time. Scope is not decided here - the plan chooses.
- Related and already promoted: `a-green-dod-proof-on-base-means-the-task-is-done`
  covers the adjacent case where a proof is green BEFORE the work starts. This
  task is about a proof that goes green without the work being done.
