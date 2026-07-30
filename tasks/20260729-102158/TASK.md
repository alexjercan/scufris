# Enrich the project task API with lifecycle and artifact metadata

- STATUS: OPEN
- PRIORITY: 64
- TAGS: feature, v0.2.0, projects, flow, backend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a Projects-page user, I want complete structured tatr metadata, so that I
can distinguish open work from history, understand `$flow` progress, and open
the records that explain a task.

## Steps

- [ ] Add failing route tests for OPEN/CLOSED status, scheduling tag/release,
      priority, Flow State, plan approval, latest review verdict, dependencies,
      and sibling artifact presence.
- [ ] Replace parsing of display-oriented `tatr ls` lines with a structured
      tatr interface or a documented Markdown parser with explicit schema
      validation and timeout/error behavior.
- [ ] Put parsing and validation behind a typed project-task reader that both
      API serializers and future server-side workflow launch guards can call;
      routes must not become the only owner of lifecycle truth.
- [ ] Add task detail and artifact-index endpoints scoped to a registered
      project's `tasks/` directory.
- [ ] Reject traversal, symlink escape, unknown artifact names, oversized
      records, and project directories outside the configured boundary.
- [ ] Represent partial or malformed historical records explicitly instead of
      silently dropping the complete task list.
- [ ] Add an integration fixture with open, closed, spike, reviewed, malformed,
      and dependency-linked tasks.

## Definition of Done

- The project task route returns lifecycle and artifact metadata for the
  integration fixture (test: `test_project_tasks_expose_flow_metadata`).
- The same typed reader returns plan approval, Flow State, review verdict, and
  dependencies without HTTP/display parsing
  (test: `test_project_task_reader_exposes_launch_guard_metadata`).
- Artifact lookup cannot escape the registered project's task directory
  (test: `test_project_task_artifact_rejects_path_escape`).
- A malformed task is reported without hiding valid siblings
  (test: `test_project_tasks_report_partial_parse_failures`).
- The route remains responsive for a fixture containing 1,000 tasks
  (test: `test_project_tasks_large_repository`).

## Notes

- Epic: 20260729-102157.
- Relevant code: `scufris/projects.py` and project routes in `scufris/app.py`.
- Record the structured tatr boundary in `DECISION.md`.
- V0.2.0 readiness role: tatr files remain authoritative; the future
  orchestrator stores assignments/observations but re-reads this boundary
  before allowing a lifecycle action.
