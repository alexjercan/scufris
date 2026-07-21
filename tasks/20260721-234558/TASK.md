# U1: orchestrator as a first-class hidden, editable agent (exclude from list, edit via settings store)

- STATUS: OPEN
- PRIORITY: 50
- TAGS: agents,backend,spike

## Goal

Make the orchestrator a first-class agent that is HIDDEN from the `/agents` list
but reachable at `/api/agents/orchestrator`, projectless, and EDITABLE through the
same per-agent path the unified settings form uses. Foundation for the shared
settings surface (Agents UX v3).

- Exclude the orchestrator from the `/api/agents` list (it is the "hidden
  default"); keep `/api/agents/orchestrator` resolving so its pages work.
- Lift the per-agent PATCH 409 for the orchestrator: route its field edits
  (backend/model/permission_mode) to the SETTINGS STORE (mapping to
  `SCUFRIS_AGENT_BACKEND`/`_MODEL`/... which the orchestrator record already
  reads), so the shared settings form edits it. Keep the rebuild-key session
  clear on a backend change.
- Keep it projectless (already `project_id=""`) and undeletable.

## Notes
- EPIC/umbrella: tasks/20260721-234126 (Agents UX v3). Spike:
  tasks/20260721-234433/SPIKE.md (recommendation B1). Foundation - do FIRST.
