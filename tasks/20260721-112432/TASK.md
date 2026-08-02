# B3: agent description field + retire the required goal from the run/create flow

- PRIORITY: 46
- TAGS: agents, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Add an optional `description` to the agent (AgentRecord/AgentCreate/AgentUpdate +
common.ts). Retire the required "goal": work is driven by chatting, so the run/
chat prompt is the message, not a stored goal. Keep `goal`/`task_id` as optional
metadata (hidden from the UI) to avoid a hard migration.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation "Description + no required goal").
- Depends on: 20260721-112429 (B1). Enables the reshaped create form (F2).

## Steps

- [x] agent_store.py: add `AgentRecord.description: str = ""`; create/update take
      `description`. Keep `goal`/`task_id` as optional metadata (default "",
      hidden from the UI) - no hard migration.
- [x] app.py: `AgentCreate`/`AgentUpdate` gain `description`; keep `goal` optional
      (the UI stops sending it). run_agent unchanged (goal override or stored goal;
      the CHAT endpoint that replaces goal-as-run-input is B4).
- [x] common.ts: `Agent.description`; `AgentCreateFields` drops `goal`, adds
      `description`.
- [x] agents-view.ts: the create form's goal textarea becomes an OPTIONAL
      `description` textarea; the detail panel "goal" row becomes "description".
- [x] Tests: description round-trips (store + API); create-form submits
      description (not goal); detail shows description; migrate the frontend
      hostile-escape test off `goal`.
- [x] Full suite + npm run ci green; close-out.

## Definition of Done

- An agent has an optional `description` that round-trips create/get/update
  (test: `agent_description_round_trips`).
- The create form collects description (not goal); the detail shows it
  (test: frontend create-form + detail tests).
- Full suite + npm run ci green.

## Close-out

What changed:
- agent_store.py: `AgentRecord.description: str = ""`; create/update take
  `description` (stripped). `goal`/`task_id` kept as optional metadata (default
  "") - no migration, just retired from the UI.
- app.py: `AgentCreate`/`AgentUpdate` gain `description`; endpoints pass it.
  `goal` stays optional; run_agent unchanged (goal override or stored goal - the
  chat replacement is B4).
- common.ts: `Agent.description`; `AgentCreateFields` drops `goal`, adds
  `description`.
- agents-view.ts: create form's goal textarea -> an OPTIONAL `description`
  textarea; the detail "goal" row -> "description". Submit sends description, not
  goal.
- Tests: description round-trip (store); frontend create-form submits description;
  detail shows it; the hostile-escape test now targets the rendered `description`.

Design:
- Kept `goal` on the record (optional, hidden) rather than dropping it, so older
  records and the run-with-goal-override path keep working - the cheapest choice
  (spike default). Work becomes chat-driven in B4/F4; goal is no longer prompted
  at create time.

Result: 256 backend (+1) + 135 frontend tests, ruff + mypy clean.

Self-reflection: small and clean. The only judgement was keep-vs-drop goal;
keeping it optional avoided a migration and a run-path change while still
retiring it from the create UX.
