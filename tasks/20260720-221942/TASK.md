# A3: create-agent-with-goal end to end (launch autonomous turn, track state)

- STATUS: OPEN
- PRIORITY: 24
- TAGS: spike,agents

## Goal

First real vertical slice of the vision: create an agent bound to a project +
goal, launch one autonomous codex-exec turn whose prompt invokes /flow scoped to
the project cwd, and track its lifecycle via the A2 status contract
(idle|running|blocked|done|error). v1 agents are read-only sandbox + long
timeout (write-access is a deferred, separately-gated phase).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 3).
- Depends on: 20260720-221929 (A1), 20260720-221935 (A2).
- Stepless direction-level task: run /plan before /work.
