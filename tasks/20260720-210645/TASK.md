# Projects backend: per-project tatr-tasks endpoint

- STATUS: OPEN
- PRIORITY: 28
- TAGS: feature,projects,backend

## Story

As the operator, I want to see a project's tatr tasks in scufris, because the
tasks ARE the specs in spec-driven development - the project page needs its task
list. tatr is directory-scoped, so a project's tasks are the tatr tasks under
its cwd.

## Steps

- [ ] Add `GET /api/projects/{id}/tasks` in `scufris/app.py`: look up the
      project, run `tatr -r <project.cwd> ls` and return the parsed tasks. Scope
      to the project's OWN `tasks/` dir: if `<cwd>/tasks` does not exist, return
      an empty list (NOT an error) rather than letting tatr walk up to a parent's
      tasks/ (confirmed: `tatr -r <dir> ls` searches upward for a `tasks/`).
- [ ] Parse each `tatr ls` line (`<path>: [PRIORITY: N, TAGS: a, b] Title`) into
      a `ProjectTask` model `{id, title, priority, tags}` where `id` is the task
      dir name (YYYYMMDD-HHMMSS from the path). Reuse the parsing approach from
      `scufris/mcp_server.py` (`tatr_ls`) if it already parses; otherwise a small
      regex. Never raise - a tatr failure/timeout returns an empty list + logs.
- [ ] 404 when the project id is unknown.
- [ ] Tests: against a temp project dir with a real `tatr new`-created task,
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
