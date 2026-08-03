# Delegate legacy /api/agent/* routes to orchestrator diagnostics

- PRIORITY: 74
- TAGS: bug, v0.2.0, agents, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102148

## Story

As an operator using the landing page, I want the legacy `/api/agent/*`
endpoints to answer from the same diagnostics service as the scoped routes, so
that switching the orchestrator away from Codex stops leaking stale Codex
account, usage, and memory data into the UI.

## Steps

- [x] Add the failing contract test
      `test_orchestrator_surfaces_are_backend_consistent` in
      `tests/test_app.py`: for an orchestrator on codex, claude, opencode and
      mock (`Settings(agent_backend=...)`, `enable_mock_backend=True`), assert
      `/api/agent/info|account|usage|memory|health` carry the same effective
      backend, model, auth mode and capability envelopes as
      `/api/agents/orchestrator/{account,usage,memory,health}`. Red today: a
      claude orchestrator reports the codex model, a codex quota and a zero
      footprint.
- [x] Add the failing delegation test
      `test_legacy_agent_routes_delegate_to_scoped_diagnostics` in
      `tests/test_app.py`: a non-codex orchestrator with a POPULATED
      `codex_home` must report `supported: false` on `/api/agent/usage` and
      `/api/agent/memory` and a `quota` of `{"supported": false, "value":
      null}` on `/api/agent/account` - the rollout data on disk must not reach
      the wire.
- [x] Move `/api/agent/usage`, `/api/agent/memory` and `/api/agent/account`
      (`scufris/app.py:3571-3603`) onto `diagnostics`, resolving the
      orchestrator record via `_require_agent(ORCHESTRATOR_ID)`. Usage and
      memory take the `Capability[T]` envelope the scoped routes already use
      (see DECISION.md); the `settings.agent_enabled` short-circuit goes, since
      the record-scoped service is the single answer and `enabled` is already
      on `AccountInfo`.
- [x] Move `/api/agent/info` (`scufris/app.py:1881`) onto the orchestrator
      record so `model` is `default_model_for(settings, backend)`, not the
      codex-only `settings.agent_model`. Same one-line fix for `_agent_config`
      (`scufris/app.py:1890`), which feeds `GET/PATCH /api/agent/config`.
- [x] Move `/api/agent/health` (`scufris/app.py:3430`) onto
      `diagnostics.health(orchestrator)` so `has_scufris_mcp` and `agent_id`
      follow the record instead of the `True`/`""` defaults.
- [x] Move the Telegram provider bundle's `usage()` and `health()`
      (`scufris/app.py:2875-2884`) onto the service, unwrapping the envelope at
      the boundary to keep the `SettingsOps` signatures. Envelope-aware
      rendering stays with 20260801-100419.
- [x] Drop the now-unused `read_usage`, `read_memory_footprint` and
      `resolve_codex_home` imports (`scufris/app.py:176-178`) and re-grep for
      direct backend-specific account reads outside `scufris/backends/`.
- [x] Adapt the one frontend consumer: `loadUsage` in `web/src/agent-view.ts:146`
      unwraps `Capability<UsageQuota>` before `renderUsage`, mirroring
      `web/src/agent-settings-view.ts`; add the envelope type in
      `web/src/agent-types.ts` if it is not already exported. `renderUsage`
      already hides the meter on a null value, so unsupported renders as
      absent, not as a zero.
- [x] Pin the deliberate divergence: `/api/agent/tools` and `/api/agent/mcp`
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

## Close-out

### What and why

The legacy singular family became compatibility ALIASES: `/api/agent/info`,
`/config`, `/account`, `/usage`, `/memory` and `/health` now resolve
`_require_agent(ORCHESTRATOR_ID)` and delegate to `AgentDiagnostics`, exactly as
their `/api/agents/orchestrator/*` twins do. Usage and memory took the
`Capability[T]` envelope per DECISION.md; `info`/`config` read `orchestrator.model`
and `orchestrator.backend` instead of the codex-only `settings.agent_model`;
`health` goes through `diagnostics.health`, so `has_scufris_mcp` follows the
record. The `settings.agent_enabled` short-circuits are gone. `/api/agent/tools`
and `/api/agent/mcp` are unchanged and the divergence is now pinned by test.

Beyond the enumerated Steps, `_run_scheduled_checks`' inner `health()` (the
digest scheduler, `scufris/app.py:2753`) was moved onto the service too. It was
the last `agent_health(settings, is_orchestrator=True)` caller in `app.py` and
carried the same `has_scufris_mcp=True` default, so leaving it would have made
the nightly digest disagree with the console it summarizes - and would have kept
`agent_health` imported into `app.py` for a stale reading.

### Alternatives

The schema-stable variant (map unsupported to null/zeros) and the
tools/mcp-mirroring variant were both rejected in DECISION.md before
implementation; nothing in the code contradicted that, so both stand.

### Difficulties

- `test_agent_health_endpoint_reports_checks` used `agent_backend=MOCK` merely to
  keep the probe deterministic, and asserted an `mcp: scufris` row. Delegation
  makes a mock orchestrator report the single "none" row instead - the intended
  behaviour change, not a regression. The test now asserts BOTH: mock gets the
  "none" row, and a codex-backed app still gets the per-server rows.
- Three disabled-agent tests reached the default (real) `codex_home` once the
  short-circuit went, which would have read the developer's own `~/.codex`. Each
  now pins an empty `codex_home` under `tmp_path`.
- `scufris/health.py:258` still calls `resolve_codex_home` for its session-count
  check. Out of scope (the DoD grep is `scufris/app.py`, and `health.py` owns its
  own probing), but it is the remaining codex-shaped read outside
  `scufris/backends/`.

### Evidence

- `python -m pytest` -> 981 passed (exit 0). Was 982 before round 1 collapsed
  two redundant disabled-agent cases into one (R1.3).
- `ruff check .` + `ruff format --check .` + `mypy .` -> clean (194 files).
- `cd web && npm run ci` -> exit 0 (format, lint, test, build).
- `rg -n "resolve_codex_home" scufris/app.py` -> empty (exit 1).

### Round 1 fixes

All five findings answered in REVIEW.md; the two load-bearing ones:

- R1.1 - the `/api/agent/usage` envelope unwrap in `loadUsage` had no frontend
  pin, because the jsdom stub answered `{}` for it. A new describe in
  `agent-view.test.ts` mounts `startAgent` over a real `#usage-meter` and both
  envelope shapes. Verified red by dropping the unwrap.
- R1.3 - the two disabled-agent tests had stopped depending on
  `agent_enabled=False`, leaving DECISION-4 unasserted. They collapse into
  `test_disabled_agent_is_supported_not_unsupported`, which asserts the property
  the decision actually claims: an account reporting `enabled: false` beside a
  `supported: true` quota.

R1.2 also applied to the MOCK-backed client in the same test, not just the
codex one the finding named: `scufris/health.py:258` gates on `agent_enabled`,
not on the backend. Both now pin an empty `codex_home`.

The review's second process signal is seeded as 20260803-032950 rather than
fixed here; the first is answered by a new row in DECISION.md's surface table.

### Reflection

The Steps' line references were all accurate, which made this mostly mechanical;
the cost was concentrated in the tests the behaviour change invalidated. Worth
noting for the sibling tasks: removing a `settings.agent_enabled` short-circuit
turns previously-inert tests into ones that touch the real home directory. Grep
for default-`Settings` construction whenever a short-circuit is deleted.
