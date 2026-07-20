# Projects concept + sesh integration (group sessions, per-project context)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,agent,spike

## Goal

A "projects" concept for the agent: group sessions by project, give each a saved
context/cwd, and integrate with `sesh` (the tmux-sessionizer that manages the
user's project dirs under ~/personal, ~/work). Let the user create/open a project
and see its sessions + a pinned context.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User: "something like being able to create projects; more tools integrations
  like sesh".
- FUZZY - start this task with a short sub-/spike: the "project" data model is
  undefined. Is a project just a cwd group (codex already records a session's cwd),
  or a saved object with pinned context/files/env/name? Decide before building.
- codex records each session's cwd in the rollout session_meta; grouping sessions
  by cwd is the cheap first cut. `sesh --create` makes a project dir; listing
  candidate dirs under ~/personal / ~/work is scriptable.
- Lower priority; do after the P40/P30 items.
