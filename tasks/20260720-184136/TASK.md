# Settings backend: config override store + gated writable endpoint

- STATUS: CLOSED
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

- [x] Add `scufris/settings_store.py`: a `SettingsStore` resolving EFFECTIVE
      settings as env-base <- persisted JSON overrides, read/written under a new
      `state_dir` Setting (`SCUFRIS_STATE_DIR`, default `~/.local/state/scufris`).
      Whitelist (`WRITABLE_KEYS`): agent_enabled, agent_backend, agent_model,
      agent_tools_enabled, agent_timeout_seconds, poll_seconds, mcp_servers.
      NOT secrets/paths. (Amended: `log_level` dropped - live-reconfiguring
      logging is out of scope; `disabled_tools` is added by task 2 to the same
      whitelist.)
- [x] Add `settings_writable: bool = True` Setting; the store raises
      `SettingsReadOnly` on any write when false.
- [x] Make config live without a restart. (Amended approach: instead of a
      `get_settings()` provider re-read per request, the store mutates the ONE
      `Settings` object IN PLACE - `validate_assignment=True` type-checks each
      setattr - so all existing `create_app` closures and the agent's per-turn
      reads see the new value with zero rewiring. Build-time keys
      (agent_enabled/agent_backend) can't be seen by in-place mutation, so the
      non-injected agent is wrapped in `AgentHandle` (implements the `Agent`
      protocol, delegates to a swappable inner) and the store's `on_change`
      rebuilds it for those keys, carrying the session across. See Notes for
      what now reads live config.)
- [x] Add `PATCH /api/agent/config`: `AgentConfigUpdate` (extra=forbid) ->
      `store.apply`; 403 (SettingsReadOnly) when read-only, 422 (UnknownSettingKey
      / ValidationError) for a bad key/value; returns the new effective config.
- [x] Extend `AgentConfig` with `writable: bool`.
- [x] Tests: `tests/test_settings_store.py` (round-trip, whitelist, gate,
      rollback, on_change, corrupt-file, AgentHandle) + endpoint tests in
      `tests/test_app.py`.
- [x] Update `.env.example` with `SCUFRIS_STATE_DIR` and
      `SCUFRIS_SETTINGS_WRITABLE`. (No other doc surface mentioned these.)

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

## Close-out (what changed and why)

- Shipped: `scufris/settings_store.py` (SettingsStore + SettingsReadOnly /
  UnknownSettingKey + WRITABLE_KEYS/REBUILD_KEYS), `AgentHandle` in
  `scufris/agent.py`, `state_dir`/`settings_writable` + `validate_assignment`
  in `config.py`, `AgentConfig.writable` + `AgentConfigUpdate` + `PATCH
  /api/agent/config` in `app.py`, `.env.example` knobs, and tests.
- Key decision - in-place mutation over a settings provider: the plan floated
  routing every read through a `get_settings()` provider (37 `agent`/`settings`
  closures in create_app). Mutating the single `Settings` object in place is
  correct here because pydantic `validate_assignment` validates each write and
  every reader already holds that object; it avoided rewiring 37 call sites and
  keeps per-turn agent reads live for free. The only reads NOT live under this
  scheme are the build-time agent selectors (agent_enabled -> Disabled vs real;
  agent_backend -> which impl), handled by `AgentHandle.rebuild` via the
  store's `on_change`.
- Live config now covers: model, tools_enabled, timeout, poll_seconds,
  mcp_servers (per-turn/config-render reads) and enabled/backend (agent rebuild).
- Difficulty - worktree import shadowing: bare `pytest` in the sprout worktree
  imported scufris from the MAIN checkout (the editable install's absolute
  path), so a NEW symbol (`AgentHandle`) was ImportError at collection even
  though mypy was green against the worktree. Fix: run `python -m pytest` from
  the worktree, which puts CWD first on sys.path so the worktree source
  shadows. Confirmed via `inspect.getfile`. Worth a lesson.
- On_change only fires for REBUILD_KEYS so a plain model/tool change does not
  needlessly rebuild the agent. Rollback in `apply` is transactional (restores
  earlier-applied keys if a later one fails validation) and does not persist a
  failed write.
- Self-reflection: the plan's "provider per request" step would have been a
  larger, riskier diff; reading the actual closure count first (37) and the
  agent's per-turn settings access made the in-place approach the clear
  correct-and-smaller choice. Good instance of grounding the plan in the code
  before following it literally.
