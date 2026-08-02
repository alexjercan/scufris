# F3: /agents/<id> detail page + per-agent settings-edit (PATCH description/mode/etc.)

- PRIORITY: 40
- TAGS: agents, frontend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The `/agents/<id>` detail page (`agent-detail.ts`): reads the id from the path,
fetches the agent + status, renders detail + a per-agent SETTINGS-edit form
(name, description, backend, permission mode) calling the existing
`PATCH /api/agents/{id}`. Extract a shared `agentFields()` builder reused by the
create form and settings. (Chat is added in F4.)

## Steps

- [x] Extract a shared `agentFields(context, initial)` builder into
      `web/src/agent-fields.ts` (name/backend/description/mode + `read()`),
      friendly backend labels, aria-labels prefixed by `context`.
- [x] Refactor `agents-view.ts` create form to use `agentFields("new agent")`
      + the create-only project picker (unchanged aria-labels/behavior).
- [x] Add an editable settings form to `agent-detail-view.ts`
      (`agentFields("agent settings", <current values>)` + a save button)
      that PATCHes `/api/agents/{id}` via an injected `save` action; keep
      project + model as read-only rows and the live status section.
- [x] Poll-guard the detail page so a status refresh never wipes an in-progress
      edit (`editingSettings()` focus guard, mirrors the list page).
- [x] Tests: `agent-fields.test.ts` (4), settings-form + prefill + save + XSS in
      `agent-detail-view.test.ts`; create-form tests unchanged and green.

## Definition of Done

- The `/agents/<id>` page renders an editable settings form prefilled with the
  agent's name/backend/description/mode
  (test: `renders a settings form prefilled with the agent's values`).
- Submitting the form PATCHes the agent with the edited values via the injected
  save action (test: `saves edited settings on submit`); a blanked name is a
  no-op (test: `does not save when the name is blanked`).
- The create form and the settings form share one field builder
  (test: `agentFields` suite; create-form tests still pass on the shared build).
- The whole web gate passes (cmd: `npm run ci` in web/).
- manual: opening `/agents/<id>` in a browser shows the form and edits persist
  across reload (e2e-verified the API slice: served shell carries agent-detail.js
  and a PATCH of the form's fields round-trips - see close-out).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F3; DoD item 5/9).
- Depends on: 20260721-112433 (F1), 20260721-112430 (B2), 20260721-112432 (B3).
- Close-out: PATCH `/api/agents/{id}` already existed (AgentUpdate: partial,
  extra=forbid), so F3 is pure frontend - the shared `agentFields` builder is
  the reuse the task asked for, and the detail page swaps its read-only
  backend/description/mode rows for the editable form (project + model stay
  read-only: project is fixed post-create, model is derived per backend).
  Description moved from a read-only text row to a textarea, so the detail
  tests assert the textarea VALUE (not textContent, which a textarea's set
  .value does not reflect). E2e (mock backend, temp state dir): served
  `/agents/<id>` carries `agent-detail.js` + `id="agent-detail"`; a PATCH of
  name/backend/description/permission_mode persisted (Before/manual ->
  After/edit) on GET. Actual DOM form submission is the batched manual check.
