# Align Telegram and the UI with orchestrator diagnostics

- PRIORITY: 73
- TAGS: bug, v0.2.0, telegram, backend, frontend
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100415

## Story

As an operator reading Telegram, I want `/settings` and `/stats` to report the
same effective backend, model, health, and quota semantics as the web UI, so
that the two surfaces never disagree about which orchestrator is running.

## Steps

Read `DECISION.md` first: D1 fixes the three-state vocabulary every step below
renders, D2 collapses the two account providers into one, D3 says why
`/settings tools` is deliberately NOT routed through the service, and D4 says
why the landing sidebar meter is left alone.

- [ ] Add the three-state vocabulary constants to `scufris/telegram/text.py`
      (`CAP_UNSUPPORTED = "not reported by the {backend} backend"`,
      `CAP_EMPTY = "nothing reported yet"`) per DECISION D1. No renderer change
      yet.
- [ ] Add the failing Telegram tests in `tests/test_telegram_render.py` and
      `tests/test_telegram.py`, parameterized over the four backends
      (`codex`, `claude`, `opencode`, `mock`) driven off the real
      `AgentDiagnostics` + `get_backend`, not a hand-written capability table:
      `test_telegram_settings_match_orchestrator_diagnostics` asserts the
      `/settings` summary and `/settings usage` bodies carry the reading the
      service returns for that backend, and
      `test_telegram_hides_codex_account_data_for_other_backends` asserts no
      percentage, window label, plan name or rollout count appears for a
      backend whose `read_usage` is `Capability.unsupported()`
      (claude/opencode/mock all are; only `codex.py:135` reads a quota).
- [ ] Carry the envelope to the renderer: add `quota: Capability[UsageQuota]`
      to `OrchestratorInfo` in `scufris/telegram/contracts.py`, delete the
      `usage` provider from `SettingsOps`, and make `render_usage` /
      `render_settings_summary` in `scufris/telegram/render.py` take the info
      (not a bare `UsageQuota | None`) and render all three D1 readings. Drop
      the `non-codex backend` wording at `render.py:311` and the bare `n/a` at
      `render.py:367`. Update `bot.py`'s `/settings` and `/settings usage`
      dispatch to the one provider.
- [ ] Rebuild `_build_telegram_settings_ops().info` in `scufris/app.py:2872`
      from `diagnostics.account(orchestrator)` plus the backend name and
      `settings.agent_permission_mode`, with the WHOLE body under
      `asyncio.to_thread` (the codex reader rglobs every rollout; R1.1 of
      20260801-100415). Remove the now-dead `usage` provider and its
      `20260801-100419` handoff comment. Leave `tools` on the in-process
      catalog per DECISION D3.
- [ ] Add the failing web test in `web/src/agent-settings-view.test.ts`:
      an `unsupported` usage or memory envelope renders the D1 sentence, a
      supported-but-empty one renders `nothing reported yet`, and neither
      renders the bare `-`.
- [ ] Make the web panels envelope-aware: `AgentSettingsData.usage` / `.memory`
      become `Capability<UsageQuota>` / `Capability<MemoryFootprint>` in
      `web/src/agent-settings-view.ts`, stop unwrapping to `?.value ?? null` at
      line 560, and render through one `capabilityText` helper so `usagePanel`
      and `memoryPanel` share the wording. Correct the stale "a later task"
      comment at line 520.
- [ ] Correct the `web/src/agent-view.ts:148` comment to cite DECISION D4
      instead of "a later task"; the unwrap itself stays.
- [ ] Extend the "The per-agent diagnostics contract" section of
      `scufris/README.md` (line 307) with a `Consuming surfaces` table naming
      the landing page, agent settings and Telegram, the D1 vocabulary each
      renders, the rule that a new surface consumes `AgentDiagnostics` and
      renders all three states, and the D3 carve-out for the console tool
      runner.
- [ ] Run the checks: `ruff check . && mypy . && python -m pytest`, and
      `npm run ci` in `web/`.

## Definition of Done

- Telegram matches the web diagnostics for every backend
  (test: `test_telegram_settings_match_orchestrator_diagnostics`).
- Telegram never prints Codex quota or rollout counts for a non-Codex
  orchestrator (test: `test_telegram_hides_codex_account_data_for_other_backends`).
- The web agent settings page tells `unsupported` apart from `nothing reported`
  instead of rendering `-` for both
  (cmd: `cd web && npx vitest run src/agent-settings-view.test.ts`).
- No Telegram module reads a backend-specific account source directly
  (cmd: `rg -n "resolve_codex_home|read_usage" scufris/telegram/`, expected
  empty). Already green on base - a regression guard, not a red proof.
- No surface still defers the third state to a later task
  (cmd: `rg -n "a later task|non-codex backend" scufris/ web/src/`, expected
  empty; red on base with three hits).
- The contract names every consuming surface and the vocabulary it renders
  (cmd: `rg -n "Consuming surfaces" -A 8 scufris/README.md`; red on base).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- The frontend gate passes (cmd: `cd web && npm run ci`).
- Backend and account information feels consistent across the landing page,
  agent settings, and Telegram (manual: user check).

## Notes

- Epic: 20260729-102145.
- Depends on the legacy-route delegation task, so all three surfaces converge
  on one service in order rather than at once.
- This task carries the epic's backend-consistency manual acceptance, since it
  is the last surface to join.

Discovered while planning:

- The original Steps 2 and 3 were stale. `20260801-100415` already left
  `scufris/telegram/` with NO backend branching and no direct account read:
  `rg -n "resolve_codex_home|read_usage" scufris/telegram/` is empty on base,
  and the providers already call `diagnostics.health` / `diagnostics.usage`.
  What survives is the two UNWRAPS that discard the envelope's third state,
  both of which name this task in their comment
  (`scufris/app.py:2893`, `web/src/agent-settings-view.ts:520`).
- The `frontend` tag is load-bearing: the web unwrap is the same defect as the
  Telegram one, and DoD's manual check spans both, so they cannot land apart
  without the surfaces disagreeing in between.
- `scufris/health.py:258` still reads a codex session count for a claude or
  opencode orchestrator on BOTH the legacy and scoped health surfaces. Out of
  scope here and already owned by 20260803-032950; `/settings health` inherits
  the fix when that lands.
- Telegram shows no memory footprint at all, so no memory work on that surface.
