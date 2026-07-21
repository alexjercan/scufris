# F6: model selection as a per-backend dropdown/autocomplete of available models

- STATUS: CLOSED
- PRIORITY: 38
- TAGS: agents,frontend,backend

## Story

User feedback (2026-07-21): the model field in the agent settings/create form is
a free-text input. It should be a DROPDOWN (or autocomplete) of the models
available for the selected backend, so you pick a valid model instead of typing.

## Steps

- [x] Backend: extend the per-backend surface with available models. Add
      `models: list[str]` to the `BackendOption` from `GET /api/agents/backends`
      (or a dedicated endpoint). Source: codex tier(s) (gpt-5.5 / gpt-5.6 if
      exposed), claude (claude-opus-4-8 / claude-sonnet-4-6 / claude-haiku-4-5),
      mock. Keep a free-text ESCAPE (autocomplete, not a hard dropdown) so an
      operator can still enter a new id.
- [x] Frontend: in `agent-fields.ts`, render the model control as a `<select>`
      or an `<input list=...>` datalist populated from the selected backend's
      models; switching the backend repopulates options AND re-defaults the value
      (keep MB1 auto-fill). Preserve an explicit override.
- [x] Tests: model options come from the backend's list; switching backend swaps
      options + default; a custom value still round-trips; backend test for the
      models field.

## Definition of Done

- The model control offers the selected backend's models and updates on backend
  change (test: options reflect the backend; switch swaps them).
- An operator can still set a model not in the list (test: custom value
  round-trips) - OR document a deliberate hard-dropdown choice.
- Full check suite green (cmd: backend `pytest -q` + web `npm run ci`).
- manual: picking a backend shows only that backend's models.

## Notes
- Depends on: MB1 (server-authoritative backends + defaults), F3 (agentFields),
  F5 (settings surface). Coordinate with F5's settings modal.
- Decide at /plan: datalist (autocomplete, keeps free-text) vs hard `<select>`.
  Recommend datalist to preserve the escape hatch.
- Relevant: scufris/app.py (BackendOption + /api/agents/backends),
  scufris/config.py (per-backend model lists), web/src/agent-fields.ts.
- Close-out: chose the DATALIST (autocomplete) over a hard `<select>` to keep
  the free-text escape hatch (an operator can still type a model outside the
  catalog). Backend: `_BACKEND_MODELS` catalog in config.py + `models_for`
  (prepends the configured default so an env-overridden model is never hidden);
  `BackendOption.models` on `GET /api/agents/backends`. Frontend: `agentFields`
  gives the model input a `<datalist>` (exposed as `fields.modelList` so callers
  append it next to `fields.model`); switching the backend swaps the suggestions
  AND re-defaults the value (MB1 behavior preserved). e2e-verified the endpoint
  returns per-backend catalogs. 271 backend + 166 frontend tests. The visible
  dropdown/typeahead is the batched manual check.
