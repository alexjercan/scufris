# Fix the record lint that reddens nix flake check

- PRIORITY: 40
- TAGS: bug, v0.2.0, records
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a maintainer, I want `nix flake check` green on master, so that the canonical
backend gate says something about the branch under test rather than about an
older task's record.

## Steps

- [x] Give the `MAX_CHANGES` Definition of Done item in
      `tasks/20260803-014401/TASK.md` a `test:`, `cmd:` or `manual:` proof.
- [x] Confirm `tatr check` is silent and `nix flake check` gets past the
      `records` check.

## Definition of Done

- The record lint is clean (cmd: `tatr check`).

## Notes

- Found while working 20260729-102148: `checks.records` fails on master, so the
  canonical gate is red before any branch touches it.
- The message is
  `20260803-014401: bad-proof-syntax: Definition of Done item has no test:, cmd:
  or manual: proof`.
- Resolved without a change here (2026-08-03). Commit `a978f1c` gave the
  `MAX_CHANGES` item a `cmd:` proof as part of 20260803-014401's own work, which
  is the whole of this task's scope. `tatr check` exits 0 and the record lint is
  clean; verified during v0.2.0 pre-flight.
