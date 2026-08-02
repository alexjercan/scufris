# Repair the two tatr-shelling project task tests

- STATUS: OPEN
- PRIORITY: 40
- TAGS: bug,tests,backlog
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT

## Story

As a Scufris maintainer, I want `python -m pytest` to be green in the dev
shell, so that a task's `cmd:` proof means what it says instead of needing a
close-out note explaining which two failures are expected.

## Steps

- [ ] Reproduce both failures against the installed `tatr` and capture the
      output shape `_TASK_LINE_RE` no longer matches
      (`scufris/projects.py:27`).
- [ ] Decide whether Scufris parses the current output or asks `tatr` for a
      stable format, and record the choice in the task.
- [ ] Fix the parser or the call, and pin the current output shape in a test
      fixture so the next drift fails at its own boundary.
- [ ] Confirm both tests still exercise the real CLI under the `needs_tatr`
      marker and still skip cleanly where it is absent.

## Definition of Done

- `test_read_project_tasks_parses_real_tatr` passes against the installed
  `tatr` (test: `test_read_project_tasks_parses_real_tatr`).
- `test_project_tasks_endpoint` passes (test: `test_project_tasks_endpoint`).
- The whole suite is green in the dev shell
  (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Found during review of 20260801-120407 (REVIEW.md R1.4). Both tests fail
  identically on `master` at `e816f46`; they shell out to the real `tatr`,
  whose `ls` output moved. They are skipped under `nix flake check`, which is
  why the drift went unnoticed.
