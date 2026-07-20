# Retro: Projects backend - per-project tatr-tasks endpoint

- TASK: 20260720-210645
- BRANCH: feature/projects-tasks
- REVIEW ROUNDS: 1 (APPROVE, no findings)

## What went well

- Probed tatr's `-r` behavior before designing (it walks UP to the nearest
  tasks/), so the endpoint gates on `<cwd>/tasks` existing - the reviewer
  verified empirically that this gate is exactly what stops a project with no
  tasks/ from showing a PARENT's tasks. The test pins that boundary directly.
- Reused the tolerant shell-out shape from mcp_server (`_run`): never raises,
  bounded timeout, resolve via shutil.which.

## What went wrong

- One test failure diagnosed and fixed: `tatr -r <dir> new` does NOT create a
  `tasks/` dir - it errors "No 'tasks' directory found in hierarchy". The test
  helper had to `mkdir <cwd>/tasks` before `tatr new`. (This same behavior is
  why the endpoint gates on the dir.)

## What to improve next time

- Remember tatr's directory model when shelling it: `-r <dir>` searches UPWARD
  for `tasks/` and never creates it - callers that want dir-scoping must create
  or gate on `<dir>/tasks` themselves.

## Action items

- [x] Added `tatr-r-walks-up-and-needs-tasks-dir` to LESSONS.md.
- No follow-up code task. PC (the Projects page) consumes these endpoints.
