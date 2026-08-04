# Projects UI: sidebar project switcher + sessions-in-project + create-from-directory

- PRIORITY: 18
- TAGS: feature, agent, ui, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Give the agent page a projects nav: a project switcher at the TOP of the
existing sidebar (above `#session-list`), so the session list becomes "sessions
in this project". Opening a project calls the backend to set the active cwd and
re-scopes the session list. Add a "create project from a directory" affordance
that offers directories under `~/personal`/`~/work` (sesh's convention;
configurable) plus an already-known cwd. Additive to the current sidebar, not a
redesign. Renders projects in the three natural kinds from the spike (saved
with sessions, auto-project by basename, empty saved project).

## Notes

- Spike: tasks/20260720-182842/SPIKE.md. Depends on the backend task
  (20260720-182938) landing first (needs the projects + active-cwd endpoints).
- Stepless: run `/plan` when picked up.
- scufris does NOT drive tmux (sesh is interactive/tty-only); reuse only its
  directory convention. Sidebar entry point: `web/src/agent-view.ts`
  `renderSessions` / `#session-list`.


- SUPERSEDED (20260720) by projects orchestrator P0 (umbrella 20260720-210347): the minimal cwd-grouping design is re-cut into the first-class Project entity. The session<->cwd scoping and per-turn -C parts here defer to P1/P2 (per-project agent). See tasks/20260720-184150/SPIKE.md Revision 1.
