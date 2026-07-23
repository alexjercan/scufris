# Decision: one registry file owns every agent's session id; agents.json stops persisting it

- DATE: 20260723-090000
- STATUS: ACCEPTED
- TASK: 20260723-001251
- TAGS: decision, agents, sessions, backend

## Context

Sub-agents persist `session_id` on their `AgentRecord` in `agents.json`; the
orchestrator holds its id in a memory-only field (`agent_store.py`
`_orch_session_id`) because it has no agents.json row (it is a synthetic record
built from settings). Two storage shapes for the same fact is exactly what let
the orchestrator's id evaporate on restart while sub-agent ids survived, and a
session id also carries an implicit backend (a codex rollout id means nothing
to claude - task 20260721-152034), which `AgentRecord.session_id` does not
record.

## Decision

Add a `SessionRegistry`: one JSON file (`<state_dir>/sessions.json`) mapping
`agent_id -> {backend, session_id}`, atomic-write/tolerant-load like the other
stores, owned and constructed by `AgentStore`. It is the ONLY persisted home of
session ids for ALL agents, orchestrator included:

- `AgentRecord.session_id` stays on the model (the API shape does not change)
  but is populated from the registry at read time, keyed by the agent's current
  backend - a backend-mismatched entry reads as None, structurally.
- `agents.json` stops persisting `session_id`; a legacy value found at load is
  migrated into the registry once (keyed by that record's backend) and then
  ignored.
- The orchestrator's `_orch_session_id` field is deleted;
  `set_orchestrator_session`/`orchestrator_session_id` delegate to the registry
  under the orchestrator's current settings backend.

## Alternatives considered

- **Persist only the orchestrator's id (small sidecar), keep AgentRecord
  persistence for sub-agents** - smallest diff, but keeps two storage shapes
  and no backend tag, so the cross-backend and delete/cleanup rules would stay
  ad-hoc per path. Rejected: the task's Direction explicitly asks for one
  first-class mapping.
- **Store the backend-tagged id back into agents.json (a `sessions` dict per
  record)** - one file, but the orchestrator still has no row there, so it
  would still need a parallel mechanism; rejected for the same asymmetry that
  caused the bug.
- **Do nothing / frontend-only guard** - leaves the restart data loss.

## Consequences

Easier: restart survival for the orchestrator's conversation; delete and
backend-switch become registry operations checked by tests; a read path can
never be handed another agent's id by accident because resolution is keyed by
`agent_id` + backend. Harder: one more state file; the pre-existing
two-store-instances pattern (app.py and mcp_server.py each build an
`AgentStore`) now also applies to sessions.json - same last-write-wins
semantics agents.json already has, accepted as the status quo.
