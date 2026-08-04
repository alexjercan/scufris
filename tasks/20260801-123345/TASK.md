# Fix the two needs_tatr project-task tests failing on master

- PRIORITY: 0
- TAGS: bug,backlog,testing
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a Scufris maintainer, I want the two `needs_tatr` project-task tests to pass
again, so that `python -m pytest` is a trustworthy gate rather than a suite with
two failures everyone has learned to ignore.

## Steps

- [ ] Reproduce on master: `python -m pytest
      tests/test_projects.py::test_read_project_tasks_parses_real_tatr
      tests/test_app.py::test_project_tasks_endpoint`. Both fail `assert 0 == 1`
      - `read_project_tasks` returns nothing after `_tatr_new` creates a task.
- [ ] Find what drifted. The tests shell out to the REAL tatr CLI (they carry
      the `needs_tatr` marker and are skipped in the nix check sandbox, which is
      why `nix flake check` stayed green while the devshell went red). Compare
      what `_tatr_new` writes and what the current `tatr ls` emits against what
      `scufris/projects.py` parses.
- [ ] Fix the parser or the fixture, whichever actually drifted, and say which
      in the close-out.
- [ ] Decide whether a CLI-version drift like this should fail loudly rather
      than skip silently in the sandbox, and record the answer.

## Definition of Done

- Both tests pass against the real CLI
  (cmd: `python -m pytest tests/test_projects.py tests/test_app.py`).
- The full suite is green in the devshell
  (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Found while reviewing 20260729-102147; pre-existing on master at 3656689 and
  unrelated to that branch, so it is filed rather than fixed there.
