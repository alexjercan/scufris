# Agent model follows backend: re-default on switch + model in settings form

- PRIORITY: 39
- TAGS: agents, bug, frontend, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, when I switch an existing agent's backend (e.g. Builder from
`mock`/`codex` to `claude`), its model must follow the new backend instead of
keeping the stale one (`gpt-5.5`). And the model should be a visible, editable
field in the per-agent settings form: changing the backend auto-fills the
backend's default model (server-authoritative), and I can still type an
override before saving.

Root cause: B1 (20260721-112429) stamped the per-backend default model only at
CREATE time (`default_model_for`); `AgentStore.update` (the PATCH path the F3
settings form uses) never recomputes `model` on a backend change - it writes
`model` only when the caller sends one (`agent_store.py:237`). So a backend
switch keeps the old model. Also `claude_model` currently defaults to `""`
(`config.py:94`), so even a correct re-default would show `-`.

## Steps

- [x] config.py: set `claude_model` default to `"claude-opus-4-8"` (the chosen
      claude default); keep it overridable via `SCUFRIS_CLAUDE_MODEL`.
- [x] agent_store.py `update()`: make `model` follow the effective backend -
      let `eff = canonical_backend(backend) if backend else agent.backend`; if
      `model is not None` use `model.strip() or default_model_for(settings,
      eff)`; elif `backend` changed the backend, set `model =
      default_model_for(settings, eff)`. (Also treat an empty create `model` as
      the default, for symmetry.)
- [x] app.py: add `GET /api/agents/backends` returning the AVAILABLE backends
      (respects `enable_mock_backend`) as `[{id, label, default_model}]` so the
      picker is server-authoritative (mock only when the flag is on) and the
      form knows each backend's default model. Add a `BackendOption` model.
- [x] common.ts: add a `BackendOption` interface + `model` to `AgentFieldValues`.
- [x] agent-fields.ts: `agentFields(context, backends: BackendOption[],
      initial)` - build the backend `<select>` from `backends`, add a `model`
      text input, and on backend `change` set `model.value` to that backend's
      `default_model` (unless the user has typed a non-default override this
      session). `read()` returns `model` too.
- [x] agents-view.ts + agent-detail-view.ts: fetch `/api/agents/backends`,
      pass it into `agentFields`; the settings form drops the read-only `model`
      row (now editable) and PATCHes `model` alongside the rest.
- [x] Tests: backend re-default on switch (unit + PATCH endpoint); the new
      endpoint shape; agent-fields model default-on-change; settings form sends
      `model`.

## Definition of Done

- Switching an agent's backend via PATCH without sending a model re-stamps the
  model to the new backend's default
  (test: `test_update_backend_redefaults_model`).
- A claude agent's default model is `claude-opus-4-8`, overridable by env
  (test: `default_model_for` claude case; `cmd: grep -n 'claude-opus-4-8' scufris/config.py`).
- `GET /api/agents/backends` lists available backends with their default models,
  and includes `mock` only when the dev flag is on
  (test: `test_agents_backends_endpoint`).
- The settings form shows an editable model field; changing the backend
  dropdown auto-fills the backend's default model
  (test: `agent-fields` change-updates-model; settings form sends `model`).
- Full check suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"` + `npm run ci` in web/).
- manual: in the browser, switch Builder mock -> claude and the model field
  updates to claude-opus-4-8 (or the configured claude model), and saving
  persists it.

## Notes
- Relevant files: scufris/config.py (default_model_for, claude_model),
  scufris/agent_store.py (update/create), scufris/app.py (AgentUpdate PATCH,
  add backends endpoint), web/src/agent-fields.ts (shared builder from F3),
  web/src/agents-view.ts, web/src/agent-detail-view.ts, web/src/common.ts.
- Depends on: F3 (20260721-112435, the shared agentFields builder + settings
  form) - landed.
- Decisions (user, 2026-07-21): model editable + auto-defaults on backend
  switch (override allowed); claude default = claude-opus-4-8.
- Close-out: two layers of defence so the model never lags the backend.
  (1) Backend `AgentStore.update` follows the EFFECTIVE backend: explicit
  non-empty model wins, a blank or omitted-model-on-backend-change re-defaults.
  This is the real API-level fix and its regression pin
  (`test_update_backend_redefaults_model` + PATCH-over-HTTP test). (2) Frontend:
  the picker + default models are now server-authoritative via
  `GET /api/agents/backends` (mock only under the dev flag), and the shared
  `agentFields` gained a `model` input that auto-fills on a backend `change`;
  the detail page's read-only model row became this editable field. Chose the
  server-driven endpoint over hardcoding defaults in the frontend so the two
  never drift (the frontend cannot know `claude_model`/`agent_model`).
  e2e (real uvicorn): `/api/agents/backends` serves unshadowed by
  `/api/agents/{id}`, and `PATCH {backend: claude}` with no model yields
  `claude / claude-opus-4-8`. Also removed the now-dead `AGENT_BACKENDS`
  frontend constant (the server list replaced it).
