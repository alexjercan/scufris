# lessons: watch format-before-check-gate + symlink for x3 promotion

- PRIORITY: 0
- TAGS: backlog, chore
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a scufris developer, I want to watch the two lessons closest to promotion
(`format-before-the-check-gate` x2 and `symlink-node_modules` x2), so that when
either hits x3 it is promoted to a guard/hook rather than lingering as prose.
Pre-emptive: fold into the guard tasks if they trip again.

## Steps

- [x] format-before-the-check-gate reviewed: still x2, recorded as a standing watch (promote at x3).
- [x] symlink-node_modules: task 20260720-220048's hooks/pre-commit now guards the commit hazard; annotated GUARDED in the ledger.
- [x] Re-ran `tatr check --ledger LESSONS.md`: clean, no promotion-stalled findings.

## Definition of Done

- Neither lesson is left stalled at x3+ without a disposition (cmd: `tatr check --ledger LESSONS.md` clean).

## Notes

- Low priority / watch item. Related: task #6 (node_modules hook).
