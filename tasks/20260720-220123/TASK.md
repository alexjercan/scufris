# tatr hygiene: clear 4 closed-unchecked lint warnings

- STATUS: OPEN
- PRIORITY: 0
- TAGS: backlog,chore

## Story

As a scufris developer, I want the 4 `closed-unchecked` lint warnings cleared,
so that `tatr check` runs clean. These are early-project CLOSED tasks with 1-2
un-checkmarked manual smoke-test steps despite the code having shipped fine.

## Steps

- [ ] Run `tatr check` and list the 4 closed-unchecked tasks.
- [ ] For each: checkmark the step if the work was genuinely done, or annotate why it was deferred/dropped.
- [ ] Confirm `tatr check` exits 0.

## Definition of Done

- `tatr check` is clean (cmd: `tatr check`).

## Notes

- Cosmetic / low severity; the code shipped correctly.
