# Projects concept + sesh integration (group sessions, per-project context)

- PRIORITY: 20
- TAGS: feature, agent, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

A "projects" concept for the agent: group sessions by project, give each a saved
context/cwd, and integrate with `sesh` (the tmux-sessionizer that manages the
user's project dirs under ~/personal, ~/work). Let the user create/open a project
and see its sessions + a pinned context.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User: "something like being able to create projects; more tools integrations
  like sesh".
- DECISION (user, 20260720): deferred - "we will need to look into this, but
  later, I think a spike is a good idea here." So this is NOT a ready feature task:
  when picked up, it OPENS WITH ITS OWN DEDICATED `/spike` to define the "project"
  concept before any building. Do not implement from this task directly.
- What the spike must decide: is a project just a cwd group (codex already records
  a session's cwd), or a saved object with pinned context/files/env/name? How does
  it relate to `sesh` (create/open project dirs under ~/personal, ~/work) and to
  the sessions list? Does it change the sidebar/nav?
- Lower priority; do after the P40/P30 items, and only once the user chooses to
  revisit it.
- SUPERSEDED (20260720): the dedicated spike is done - tasks/20260720-182842/
  SPIKE.md. It defined a project as a working directory (Option A hybrid) and
  seeded the implementation tasks 20260720-182938 (backend), -182953 (UI) and
  -182959 (pinned context). Closing this trigger task; work continues there.
