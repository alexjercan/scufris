# Repurpose config profiles into per-project agent config (projects orchestrator P1)

- STATUS: OPEN
- PRIORITY: 13
- TAGS: feature,agent,projects,spike

## Goal

Fold the landed config-profiles feature (backend store from tasks/20260720-184138,
switcher UI from tasks/20260720-184149) INTO the projects-orchestrator concept:
a "profile" becomes a PER-PROJECT agent config, not a free-standing global
switcher. The user does not want global profile switching (that role is `login`);
the config-switching axis is per project. So the profile machinery is repurposed,
not deleted.

Direction (for `/plan` to break down when picked up):
- Reuse the profile store's override machinery, but key it by PROJECT (an agent
  config belongs to a project, not a global "active profile").
- Retire the global profile SWITCHER UI (tasks/20260720-184149) in favour of a
  project's agent-config panel on the Project page.
- Keep the writable-config API (tasks/20260720-184136) as the config mechanism;
  scope it per project.

## Notes

- Spike: tasks/20260720-184150/SPIKE.md "Revision 1", phase P1. Read it first.
- User decision (20260720): REPURPOSE profiles into per-project agent config
  (not remove, not leave as-is).
- Depends on the projects P0 work (a first-class Project entity + page) landing
  first - the per-project keying has nowhere to live until projects exist.
- Stepless: run `/plan` when picked up (and confirm how much of the current
  profiles store/endpoints survive vs is replaced).

