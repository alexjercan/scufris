# A1: AgentStore - agent as a first-class record (agents.json + CRUD)

- STATUS: OPEN
- PRIORITY: 28
- TAGS: spike,agents

## Goal

Make "agent" a first-class entity. Add an `AgentStore` persisting `agents.json`
(mirroring projects.py / settings_store.py: atomic write, tolerant load, gated
by settings_writable) with records `{id, name, project_cwd, backend, model,
goal|task_id, session_id, state}` and CRUD API. Demote Project from a
destination page to the project-picker plumbing behind agent creation (keep
projects.py as the data layer + tatr-tasks endpoint).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 1).
- Depends on: 20260720-221922 (A0).
- Stepless direction-level task: run /plan before /work.
