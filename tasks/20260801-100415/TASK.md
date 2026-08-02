# Delegate legacy /api/agent/* routes to orchestrator diagnostics

- PRIORITY: 74
- TAGS: bug, v0.2.0, agents, backend
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102148

## Story

As an operator using the landing page, I want the legacy `/api/agent/*`
endpoints to answer from the same diagnostics service as the scoped routes, so
that switching the orchestrator away from Codex stops leaking stale Codex
account, usage, and memory data into the UI.

## Steps

- [ ] Add the failing contract test
      `test_orchestrator_surfaces_are_backend_consistent` in
      `tests/test_app.py`: for an orchestrator on codex, claude, opencode and
      mock (`Settings(agent_backend=...)`, `enable_mock_backend=True`), assert
      `/api/agent/info|account|usage|memory|health` carry the same effective
      backend, model, auth mode and capability envelopes as
      `/api/agents/orchestrator/{account,usage,memory,health}`. Red today: a
      claude orchestrator reports the codex model, a codex quota and a zero
      footprint.
- [ ] Add the failing delegation test
      `test_legacy_agent_routes_delegate_to_scoped_diagnostics` in
      `tests/test_app.py`: a non-codex orchestrator with a POPULATED
      `codex_home` must report `supported: false` on `/api/agent/usage` and
      `/api/agent/memory` and a `quota` of `{"supported": false, "value":
      null}` on `/api/agent/account` - the rollout data on disk must not reach
      the wire.
- [ ] Move `/api/agent/usage`, `/api/agent/memory` and `/api/agent/account`
      (`scufris/app.py:3571-3603`) onto `diagnostics`, resolving the
      orchestrator record via `_require_agent(ORCHESTRATOR_ID)`. Usage and
      memory take the `Capability[T]` envelope the scoped routes already use
      (see DECISION.md); the `settings.agent_enabled` short-circuit goes, since
      the record-scoped service is the single answer and `enabled` is already
      on `AccountInfo`.
- [ ] Move `/api/agent/info` (`scufris/app.py:1881`) onto the orchestrator
      record so `model` is `default_model_for(settings, backend)`, not the
      codex-only `settings.agent_model`. Same one-line fix for `_agent_config`
      (`scufris/app.py:1890`), which feeds `GET/PATCH /api/agent/config`.
- [ ] Move `/api/agent/health` (`scufris/app.py:3430`) onto
      `diagnostics.health(orchestrator)` so `has_scufris_mcp` and `agent_id`
      follow the record instead of the `True`/`""` defaults.
- [ ] Move the Telegram provider bundle's `usage()` and `health()`
      (`scufris/app.py:2875-2884`) onto the service, unwrapping the envelope at
      the boundary to keep the `SettingsOps` signatures. Envelope-aware
      rendering stays with 20260801-100419.
- [ ] Drop the now-unused `read_usage`, `read_memory_footprint` and
      `resolve_codex_home` imports (`scufris/app.py:176-178`) and re-grep for
      direct backend-specific account reads outside `scufris/backends/`.
- [ ] Adapt the one frontend consumer: `loadUsage` in `web/src/agent-view.ts:146`
      unwraps `Capability<UsageQuota>` before `renderUsage`, mirroring
      `web/src/agent-settings-view.ts`; add the envelope type in
      `web/src/agent-types.ts` if it is not already exported. `renderUsage`
      already hides the meter on a null value, so unsupported renders as
      absent, not as a zero.
- [ ] Pin the deliberate divergence: `/api/agent/tools` and `/api/agent/mcp`
      stay the operator console's in-process ORCHESTRATOR-audience surfaces
      (they read no Codex account and already share the service's helpers).
      Assert in the contract test that they keep listing tools for a backend
      whose scoped route reports `supported: false`.

## Definition of Done

- Legacy and scoped orchestrator surfaces agree for every backend
  (test: `test_orchestrator_surfaces_are_backend_consistent`).
- No legacy route constructs Codex-only account data on its own
  (test: `test_legacy_agent_routes_delegate_to_scoped_diagnostics`).
- Codex account readers are confined to the diagnostics service
  (cmd: `rg -n "resolve_codex_home" scufris/app.py`, expected empty).
- Public route contracts do not drift
  (cmd: `python -m pytest && cd web && npm run ci`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the diagnostics service task.
- Keep the compatibility routes. They become adapters, not independent
  implementations.
- The orchestrator record is synthetic and derived from settings
  (`scufris/agent_store/reserved.py:43`), so `_require_agent(ORCHESTRATOR_ID)`
  cannot 404 and needs no new persistence.
- Wire-shape change: `/api/agent/usage` and `/api/agent/memory` gain the
  `Capability[T]` envelope. Rationale and the rejected schema-stable variant are
  in DECISION.md. Existing tests at `tests/test_app.py:1816-1905` are updated
  for the envelope.
- Behaviour change: a DISABLED orchestrator no longer short-circuits usage and
  memory; the backend reader answers and `enabled: false` on `AccountInfo` says
  why. This is what the scoped routes already do.
- `/api/agent/tools` and `/api/agent/mcp` are NOT stale-Codex leaks; the task
  description grouped them with the account family. See DECISION.md.
- No new gate or mode is introduced, so no new-gate grep applies.
