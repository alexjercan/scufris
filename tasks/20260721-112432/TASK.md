# B3: agent description field + retire the required goal from the run/create flow

- STATUS: OPEN
- PRIORITY: 46
- TAGS: agents,backend


## Goal

Add an optional `description` to the agent (AgentRecord/AgentCreate/AgentUpdate +
common.ts). Retire the required "goal": work is driven by chatting, so the run/
chat prompt is the message, not a stored goal. Keep `goal`/`task_id` as optional
metadata (hidden from the UI) to avoid a hard migration.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation "Description + no required goal").
- Depends on: 20260721-112429 (B1). Enables the reshaped create form (F2).
