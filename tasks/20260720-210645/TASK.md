# Projects backend: per-project tatr-tasks endpoint

- PRIORITY: 28
- TAGS: feature, projects, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want to see a project's tatr tasks in scufris, because the
tasks ARE the specs in spec-driven development - the project page needs its task
list. tatr is directory-scoped, so a project's tasks are the tatr tasks under
its cwd.

## Steps

- [x] Add `GET /api/projects/{id}/tasks` in `scufris/app.py`: look up the
      project, run `tatr -r <project.cwd> ls` and return the parsed tasks. Scope
      to the project's OWN `tasks/` dir: if `<cwd>/tasks` does not exist, return
      an empty list (NOT an error) rather than letting tatr walk up to a parent's
      tasks/ (confirmed: `tatr -r <dir> ls` searches upward for a `tasks/`).
- [x] Parse each `tatr ls` line (`<path>: [PRIORITY: N, TAGS: a, b] Title`) into
      a `ProjectTask` model `{id, title, priority, tags}` where `id` is the task
      dir name (YYYYMMDD-HHMMSS from the path). Reuse the parsing approach from
      `scufris/mcp_server.py` (`tatr_ls`) if it already parses; otherwise a small
      regex. Never raise - a tatr failure/timeout returns an empty list + logs.
- [x] 404 when the project id is unknown.
- [x] Tests: against a temp project dir with a real `tatr new`-created task,
      the endpoint returns that task (id/title/priority/tags); a project whose
      cwd has no `tasks/` returns `[]`; unknown project -> 404. Space multiple
      `tatr new` calls by >1s (`tatr-ids-are-second-resolution`).

## Definition of Done

- The endpoint returns a project's tatr tasks parsed into structured records
  (test: `project_tasks_endpoint`).
- A project dir with no `tasks/` returns an empty list, not an error, and does
  not walk up to a parent's tasks (test: `project_tasks_empty_when_no_tasks_dir`).
- Unknown project id -> 404 (test: `project_tasks_unknown_404`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && python -m pytest -q"`).

## Notes

- Depends on: 20260720-210644 (the Project store + CRUD).
- Relevant files: `scufris/mcp_server.py` (`tatr_ls`, `_run` - the existing tatr
  shell-out + timeout pattern), `scufris/app.py`.
- Grounding: `tatr -r <dir> ls` walks UP to the nearest `tasks/`; to keep a
  project's list to ITS tasks, gate on `<cwd>/tasks` existing before shelling.
- Lessons: `tatr-ids-are-second-resolution` (space test `tatr new`s);
  `set-e-plus-grep-c-aborts-scripts` (bare shell-out, check the file/exit).

## Close-out

- Added `ProjectTask` + `read_project_tasks(cwd)` to `scufris/projects.py`
  (cohesive with the store) and `GET /api/projects/{id}/tasks` in `app.py`.
- The key correctness point: `read_project_tasks` gates on `<cwd>/tasks`
  existing and returns `[]` WITHOUT calling tatr otherwise, so tatr's `-r`
  upward search can never surface a PARENT's tasks for a project that has none
  of its own. Pinned by `test_read_project_tasks_empty_when_no_tasks_dir` (a
  child dir under a parent that DOES have tasks/ returns []).
- Never-raises: a missing tatr, timeout (10s cap) or non-zero exit -> [] + log,
  mirroring `mcp_server._run`'s tolerant shell-out.
- Parse: one regex over `tatr ls` lines (`<path>: [PRIORITY: N, TAGS: ...]
  Title`); id = the task dir name (parent of TASK.md).
- Test gotcha found: `tatr -r <dir> new` does NOT create a `tasks/` dir - it
  errors "No 'tasks' directory found in hierarchy". So the test helpers mkdir
  `<cwd>/tasks` before `tatr new`. (This is also why the endpoint gates on the
  dir - a real project needs a tasks/ for tatr to work.)
- 203 backend tests pass; `python -m pytest` from the worktree.
