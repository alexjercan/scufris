# U1: orchestrator as a first-class hidden, editable agent (exclude from list, edit via settings store)

- STATUS: CLOSED
- PRIORITY: 50
- TAGS: agents, backend, spike
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

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

## Steps (/plan)

- [x] `config.py`: add `agent_permission_mode: PermissionMode = MANUAL` (env
      `SCUFRIS_AGENT_PERMISSION_MODE`) for the orchestrator's write posture; doc
      it in `.env.example`. `settings_store.py`: add `claude_model` and
      `agent_permission_mode` to `WRITABLE_KEYS` (so the orchestrator's model +
      mode are runtime-editable); `agent_permission_mode` is NOT a rebuild key.
- [x] `agent_store.py`: `_orchestrator_record` reads `permission_mode` from
      `settings.agent_permission_mode` (was hardcoded MANUAL); model already
      follows `default_model_for(backend)`. `list()` EXCLUDES the reserved
      orchestrator (returns only the real agents); `get(ORCHESTRATOR_ID)` still
      resolves. Add `list(include_reserved=False)` only if a caller needs the old
      behavior (mcp_server agent-list should also hide it - "hidden default").
- [x] `app.py` `PATCH /api/agents/{id}` (`update_agent`): for `ORCHESTRATOR_ID`,
      translate the `AgentUpdate` fields to settings-store keys - backend ->
      `agent_backend`, model -> `agent_model` (codex) or `claude_model` (claude)
      by the effective backend, permission_mode -> `agent_permission_mode` - and
      apply via the `SettingsStore` (SettingsReadOnly -> 403, invalid -> 422),
      returning the refreshed `agents.get(ORCHESTRATOR_ID)`. The existing
      `_on_settings_change` still clears the orchestrator session on a backend
      change. Non-orchestrator agents keep the current `AgentStore.update` path.
- [x] Tests: `/api/agents` list EXCLUDES the orchestrator while
      `/api/agents/orchestrator` still resolves; `PATCH /api/agents/orchestrator`
      changes backend/model/permission_mode through the settings store (persists +
      reflects in the record; backend change clears its session); read-only -> 403;
      a project agent PATCH is unchanged; orchestrator still projectless +
      undeletable. Update the reverse "synthetic is first in the list" assertions
      (lesson `always-present-synthetic-item-invalidates-empty-assertions`).
- [x] Full check suite green (ruff + mypy + pytest; web `npm run ci` for any
      frontend fallout - the agents-view orchestrator handling may need a touch).

## Definition of Done

- `GET /api/agents` does NOT include the orchestrator; `GET /api/agents/
  orchestrator` resolves (test: list excludes it, get resolves).
- `PATCH /api/agents/orchestrator` edits backend/model/permission_mode via the
  settings store and they persist + reflect in the orchestrator record; a
  read-only server returns 403 (test).
- The orchestrator stays projectless (`project_id == ""`) and undeletable (403)
  (test unchanged).
- Full check suite green.

## Notes
- EPIC/umbrella: tasks/20260721-234126 (Agents UX v3). Spike:
  tasks/20260721-234433/SPIKE.md (recommendation B1). Foundation - do FIRST.
- Design: the orchestrator's config is settings-store-backed (it has no
  agents.json row), so its "edit" writes settings keys; a project agent's edit
  writes its record. The unified settings FORM (U3) is identical either way; only
  the endpoint's dispatch differs.
