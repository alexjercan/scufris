# U4: routing/entries so / == orchestrator and /agents/<id>[/settings] share the components

- STATUS: OPEN
- PRIORITY: 44
- TAGS: agents,frontend,spike

## Goal

Wire the routing so `/` and `/settings` are the ORCHESTRATOR's chat + settings and
`/agents/<id>` and `/agents/<id>/settings` are a project agent's - all using the
SAME components, differing only by the resolved agent id.

- `/settings` entry mounts the unified settings (U3) with agent id =
  `orchestrator`; `/agents/<id>/settings` mounts it with `agentIdFromPath` (the
  backend `/agents/{id}/{rest:path}` catch-all already serves the shell - confirm,
  add the entry/mount that reads the `/settings` sub-path).
- `/` and `/agents/<id>` already share the chat (B5d) - confirm nothing regresses.
- Retire the per-agent settings modal + its toggle now that settings is a page.
- Add a "Settings" affordance on each agent's page linking to
  `/agents/<id>/settings`.

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation A1 / U4). Depends on U3.
