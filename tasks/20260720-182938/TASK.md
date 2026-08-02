# Projects backend: cwd-scoped sessions, project store, per-turn working-root

- PRIORITY: 20
- TAGS: feature, agent, backend, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Make the agent backend project-aware, where a "project" is a working directory
(see the spike for the model). Deliver: (1) a tiny `projects.json` store under
scufris state holding only `[{cwd, name, context_md}]`; (2) replace the
hardcoded `list_sessions(home, os.getcwd())` filter with an ACTIVE-project cwd,
so the session list is scoped to the open project's cwd (keeping the
`_SCUFRIS_ORIGINATORS` scope); (3) an endpoint listing projects = union of
distinct session cwds + saved records; (4) pass `-C <cwd>` to `codex exec` (and
the app-server equivalent) so a turn runs in the project's working root;
(5) validate/confine the cwd (allowlist root by default - see open question).
Membership stays derived from codex's recorded cwd - do NOT add a per-session
project_id (avoid a second source of truth).

## Notes

- Spike: tasks/20260720-182842/SPIKE.md (Option A hybrid). Read it first.
- Stepless: run `/plan` when picked up to break this into steps.
- Open questions the spike leaves for build time: cwd allowlist vs free-form;
  whether `codex app-server` `thread/start` accepts a working root (probe - the
  exec path can ship first).


- SUPERSEDED (20260720) by projects orchestrator P0 (umbrella 20260720-210347): the minimal cwd-grouping design is re-cut into the first-class Project entity. The session<->cwd scoping and per-turn -C parts here defer to P1/P2 (per-project agent). See tasks/20260720-184150/SPIKE.md Revision 1.
