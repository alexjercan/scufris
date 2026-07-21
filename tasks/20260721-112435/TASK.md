# F3: /agents/<id> detail page + per-agent settings-edit (PATCH description/mode/etc.)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: agents,frontend


## Goal

The `/agents/<id>` detail page (`agent-detail.ts`): reads the id from the path,
fetches the agent + status, renders detail + a per-agent SETTINGS-edit form
(name, description, backend, permission mode) calling the existing
`PATCH /api/agents/{id}`. Extract a shared `agentFields()` builder reused by the
create form and settings. (Chat is added in F4.)

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F3; DoD item 5/9).
- Depends on: 20260721-112433 (F1), 20260721-112430 (B2), 20260721-112432 (B3).
