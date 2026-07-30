# tatr hygiene: clear 4 closed-unchecked lint warnings

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: backlog, chore
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As a scufris developer, I want the 4 `closed-unchecked` lint warnings cleared,
so that `tatr check` runs clean. These are early-project CLOSED tasks with 1-2
un-checkmarked manual smoke-test steps despite the code having shipped fine.

## Steps

- [x] Run `tatr check` and list the 4 closed-unchecked tasks.
- [x] For each: ticked (work verified done via RETRO/code) with a transparent hygiene-pass annotation.
- [x] Confirm `tatr check` exits 0.

## Definition of Done

- `tatr check` is clean (cmd: `tatr check`).

## Notes

- Cosmetic / low severity; the code shipped correctly.
