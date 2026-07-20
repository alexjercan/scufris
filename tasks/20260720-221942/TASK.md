# A3: create-agent-with-goal end to end (background job, gated write, tracked state)

- STATUS: OPEN
- PRIORITY: 24
- TAGS: spike,agents

## Goal

First real vertical slice of the vision: create an agent bound to a project +
goal, with the per-agent, cwd-scoped **write** opt-in (decision 3); launch it as
a **background job** via the A0 supervisor (no held request, no timeout); its
prompt invokes /flow scoped to the project cwd; track its lifecycle
(idle|running|blocked|done|error) via the A2 status contract, surfaced by
polling.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 3; decisions 2,3).
- Depends on: 20260720-221929 (A1), 20260720-221935 (A2).
- Stepless direction-level task: run /plan before /work.
