# Goal: promote scufris flow footgun lessons into guards + clear tatr lint

- DATE: 20260720
- UMBRELLA TASK: 20260720-225502
- LANDING SCOPE: squash-merge each task to local master; do NOT push (user's call).

## Goal

Turn scufris's recurring flow footguns (recorded in LESSONS.md) from prose into
actual guards, and clear the outstanding tatr lint. Two guards: a pre-commit
hook that refuses a staged `web/node_modules` symlink (which corrupted a branch
twice), and a pytest/conftest guard that refuses to run tests importing scufris
from outside the current worktree (bare `pytest` in a sprout silently imports
the main checkout). Plus two cleanups: clear the 4 `closed-unchecked` tatr
findings, and record the promotion disposition for the two x2 watch-lessons.

## Done means

1. Staging `web/node_modules` fails the commit with a clear message; a normal commit still succeeds (manual: attempt both).
2. Running tests that import scufris from outside the current worktree fails fast with a pointer to `python -m pytest` (manual: reproduce in a sprout).
3. `tatr check` is clean - the 4 closed-unchecked findings are resolved (cmd: `tatr check`).
4. Neither x2 watch-lesson is left without a disposition; ledger lints clean (cmd: `tatr check --ledger LESSONS.md`).

Overall: `nix flake check` green; `tatr check` and `tatr check --ledger LESSONS.md` clean.

## Tasks

- [x] 20260720-220048 (p0) pre-commit hook: reject web/node_modules symlink
      landed 7fc2ed0; 1 review round (APPROVE, no findings). NOTE: nix flake
      check pytest is pre-existing red on master (No module named scufris);
      verified via devShell (ruff+mypy+pytest 203 passed). Follow-up task to file.
- [x] 20260720-220101 (p0) worktree pytest guard: enforce python -m pytest
      landed 19a48f4; 2 review rounds (APPROVE r1 with 1 NIT; NIT fixed - subdir
      false-fire). conftest fails fast if scufris imports from outside cwd.
- [ ] 20260720-220123 (p0) tatr hygiene: clear 4 closed-unchecked lint warnings
- [ ] 20260720-220116 (p0) lessons: disposition format-before-check-gate + symlink

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.
