# A1: AgentStore - agent as a first-class record (agents.json + CRUD)

- STATUS: OPEN
- PRIORITY: 28
- TAGS: spike,agents

## Goal

Make "agent" a first-class entity. Add an `AgentStore` persisting `agents.json`
(mirroring projects.py / settings_store.py: atomic write, tolerant load, gated
by settings_writable) with records `{id, name, project_cwd, backend, model,
goal|task_id, session_id, state, write_enabled}` and CRUD API. `backend` selects
codex vs claude (the common interface, A2/A2b); `write_enabled` is the per-agent,
cwd-scoped write opt-in (decision 3); `state` is the lifecycle
(idle|running|blocked|done|error). Demote Project from a destination page to the
project-picker plumbing behind agent creation (keep projects.py as the data
layer + tatr-tasks endpoint).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 1; decisions 1,3).
- Depends on: 20260720-221922 (A0).
- Stepless direction-level task: run /plan before /work.
