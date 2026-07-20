# Review: Projects backend - per-project tatr-tasks endpoint

- TASK: 20260720-210645
- BRANCH: feature/projects-tasks

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, ran the full nix suite + verified
  tatr walk-up behavior empirically)

No BLOCKER/MAJOR/MINOR/NIT findings. Full suite green (ruff + mypy + pytest via
`python -m pytest`, 203 tests). Reviewer confirmed empirically that
`tatr -r <child> ls` DOES walk up to a parent's tasks, and that the
`<cwd>/tasks` gate is what prevents it - `test_read_project_tasks_empty_when_no_
tasks_dir` genuinely fails if the gate is removed. All never-raise paths
(tatr-not-on-PATH, 10s timeout, non-zero exit, SubprocessError/OSError) return
`[]` + log; the regex matches real `tatr ls` output including the empty-tags
case; the id is the correct task dir name; security is sound (explicit argv,
shell=False, shutil.which, operator-created cwd); 404 on unknown project is
tested. Every DoD test uses real `tatr` and fails if its mechanism is removed.

No open `manual:` DoD items.
