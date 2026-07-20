# lessons: watch format-before-check-gate + symlink for x3 promotion

- STATUS: OPEN
- PRIORITY: 0
- TAGS: backlog,chore

## Story

As a scufris developer, I want to watch the two lessons closest to promotion
(`format-before-the-check-gate` x2 and `symlink-node_modules` x2), so that when
either hits x3 it is promoted to a guard/hook rather than lingering as prose.
Pre-emptive: fold into the guard tasks if they trip again.

## Steps

- [ ] If `format-before-the-check-gate` recurs (x3), add `ruff format` ahead of the check gate in the shared check command / pre-commit.
- [ ] If `symlink-node_modules` recurs, ensure task #6's hook covers it (this task then folds into #6).
- [ ] Re-run `tatr check --ledger LESSONS.md` to confirm no promotion-stalled findings.

## Definition of Done

- Neither lesson is left stalled at x3+ without a disposition (cmd: `tatr check --ledger LESSONS.md` clean).

## Notes

- Low priority / watch item. Related: task #6 (node_modules hook).
