# Settings backend: config override store + gated writable endpoint

- STATUS: OPEN
- PRIORITY: 45
- TAGS: feature,agent,backend,config

## Story

As the homelab operator, I want the settings I see on the page to be
changeable at runtime and to STICK across restarts, so I can tune the agent
without editing `.env` and rebooting the service. Env stays the first-boot
seed; a small persisted override layer sits on top.

This is the foundation task for the interactive settings console
(umbrella 20260720-183719): the override store + a gated write endpoint that
every later task builds on.

## Steps

- [ ] Add `scufris/settings_store.py`: a `SettingsStore` that resolves the
      EFFECTIVE settings as `env-base Settings()` <- persisted JSON overrides,
      reading/writing an overrides file under a scufris state dir (add a
      `state_dir` Setting, default e.g. `~/.local/state/scufris`, env
      `SCUFRIS_STATE_DIR`). Only whitelisted, safe-to-mutate keys may be
      overridden (agent_enabled, agent_backend, agent_model,
      agent_tools_enabled, poll_seconds, log_level, mcp_servers, and the
      new disabled-tools field from task 2) - NOT secrets/paths
      (openai_api_key, codex_bin, codex_home, host, port).
- [ ] Add a `settings_writable: bool = True` Setting (env
      `SCUFRIS_SETTINGS_WRITABLE`); when false the store rejects writes.
- [ ] Make the app read effective settings through the store per request
      instead of the settings captured once in `create_app` closures: inject
      the store (or a `get_settings()` provider) so a mutation is visible to
      the next request without a restart. Grep every closure over `settings`
      in `scufris/app.py` and route reads through the provider; list in Notes
      what newly reads live config.
- [ ] Add `PATCH /api/agent/config` (or `POST /api/agent/config`): validate the
      partial update against the whitelist, apply+persist via the store, return
      the new effective `AgentConfig`. Return a clear 403 (not 500) when
      `settings_writable` is false.
- [ ] Extend `AgentConfig` with a `writable: bool` field so the frontend knows
      whether to render controls.
- [ ] Tests: store round-trip (write override -> new `Settings()` in a fresh
      process/tmp state dir reflects it, env-unset keys keep env values);
      endpoint applies + persists; endpoint returns 403 when the gate is off;
      a non-whitelisted key is rejected.
- [ ] Update `.env.example` and any config doc with `SCUFRIS_STATE_DIR` and
      `SCUFRIS_SETTINGS_WRITABLE`.

## Definition of Done

- A persisted override survives a restart: writing an override then building a
  fresh effective settings reads it back, while env-only keys keep env values
  (test: `settings_store_round_trip`).
- The write endpoint applies and persists a whitelisted change and returns the
  new effective config (test: `patch_agent_config_persists`).
- With `settings_writable=false` the endpoint refuses with 403 and a clear
  message (test: `patch_agent_config_forbidden_when_readonly`).
- A non-whitelisted key (e.g. `openai_api_key`) is rejected, not written
  (test: `patch_agent_config_rejects_non_whitelisted`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).

## Notes

- Relevant files: `scufris/config.py` (Settings), `scufris/app.py`
  (`get_agent_config` at ~245, all `create_app` closures over `settings`),
  `scufris/agent.py` (consumes settings per turn - must see live values).
- Design: a "profile" (task 3) is a NAMED override set; keep the store's
  on-disk shape profile-ready (e.g. `{active, profiles:{default:{...}}}`) so
  task 3 is additive, not a rewrite. Decide this shape here.
- pydantic-settings loads env at `Settings()` construction; the store composes
  a base `Settings()` with overrides via `model_copy(update=...)` on the
  whitelisted keys rather than re-parsing env.
- Assumption (user 20260720): writable defaults ON (single-operator local
  tool); safety is the whitelist + the gate + a UI confirm (task 5), not a
  default-off.
