# Align Telegram and the UI with orchestrator diagnostics

- PRIORITY: 73
- TAGS: bug, v0.2.0, telegram, backend, frontend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
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

- [x] Add the three-state vocabulary constants to `scufris/telegram/text.py`
      (`CAP_UNSUPPORTED = "not reported by the {backend} backend"`,
      `CAP_EMPTY = "nothing reported yet"`) per DECISION D1. No renderer change
      yet.
- [x] Add the failing Telegram tests in `tests/test_telegram_app.py` (see the
      close-out; NOT `test_telegram.py`/`test_telegram_render.py`), parameterized over the four backends
      (`codex`, `claude`, `opencode`, `mock`) driven off the real
      `AgentDiagnostics` + `get_backend`, not a hand-written capability table:
      `test_telegram_settings_match_orchestrator_diagnostics` asserts the
      `/settings` summary and `/settings usage` bodies carry the reading the
      service returns for that backend, and
      `test_telegram_hides_codex_account_data_for_other_backends` asserts no
      percentage, window label, plan name or rollout count appears for a
      backend whose `read_usage` is `Capability.unsupported()`
      (claude/opencode/mock all are; only `codex.py:135` reads a quota).
- [x] Carry the envelope to the renderer: add `quota: Capability[UsageQuota]`
      to `OrchestratorInfo` in `scufris/telegram/contracts.py`, delete the
      `usage` provider from `SettingsOps`, and make `render_usage` /
      `render_settings_summary` in `scufris/telegram/render.py` take the info
      (not a bare `UsageQuota | None`) and render all three D1 readings. Drop
      the `non-codex backend` wording at `render.py:311` and the bare `n/a` at
      `render.py:367`. Update `bot.py`'s `/settings` and `/settings usage`
      dispatch to the one provider.
- [x] Rebuild `_build_telegram_settings_ops().info` in `scufris/app.py:2872`
      from `diagnostics.account(orchestrator)` plus the backend name and
      `settings.agent_permission_mode`, with the WHOLE body under
      `asyncio.to_thread` (the codex reader rglobs every rollout; R1.1 of
      20260801-100415). Remove the now-dead `usage` provider and its
      `20260801-100419` handoff comment. Leave `tools` on the in-process
      catalog per DECISION D3.
- [x] Add the failing web test in `web/src/agent-settings-view.test.ts`:
      an `unsupported` usage or memory envelope renders the D1 sentence, a
      supported-but-empty one renders `nothing reported yet`, and neither
      renders the bare `-`.
- [x] Make the web panels envelope-aware: `AgentSettingsData.usage` / `.memory`
      become `Capability<UsageQuota>` / `Capability<MemoryFootprint>` in
      `web/src/agent-settings-view.ts`, stop unwrapping to `?.value ?? null` at
      line 560, and render through one `capabilityText` helper so `usagePanel`
      and `memoryPanel` share the wording. Correct the stale "a later task"
      comment at line 520.
- [x] Correct the `web/src/agent-view.ts:148` comment to cite DECISION D4
      (it did not in fact say "a later task"; it explained the hiding without a
      reason). The unwrap itself stays.
- [x] Extend the "The per-agent diagnostics contract" section of
      `scufris/README.md` (line 307) with a `Consuming surfaces` table naming
      the landing page, agent settings and Telegram, the D1 vocabulary each
      renders, the rule that a new surface consumes `AgentDiagnostics` and
      renders all three states, and the D3 carve-out for the console tool
      runner.
- [x] Run the checks: `ruff check . && mypy . && python -m pytest`, and
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

## Close-out

WHAT. Both surfaces now read the `Capability` envelope's three states instead of
collapsing two of them. `OrchestratorInfo` carries `quota: Capability[UsageQuota]`
and `SettingsOps.usage` is gone, so `/settings` and `/settings usage` answer from
the one `diagnostics.account()` call `info()` makes (D2). `render.py`'s
`_quota_reading` and `agent-settings-view`'s `capabilityText` are the two
language-local copies of the D1 vocabulary, and the new `Consuming surfaces`
section of `scufris/README.md` is what they are copies of.

WHY. `no usage data (agent disabled or non-codex backend)` and a bare `-` both
read as breakage to a claude operator, when the truth is that only `codex.py`
has a quota reader at all.

DEVIATIONS.

- The two new Telegram tests live in `tests/test_telegram_app.py`, not in the
  two files the Steps named. They boot the REAL app per backend and drive the
  real `/settings` commands, comparing the printed body against what
  `AgentDiagnostics` actually returns - which is the point of "not a
  hand-written capability table", and which neither the pure-render module nor
  the fake-driven transport module can do without importing the app into them.
  `test_telegram_render.py` did get the pure three-state render tests.
- `web/src/agent-view.ts` never said "a later task"; the comment now cites D4.
- `web/src/agent-settings-view.ts` was at 593 of its 600-line cap, so this
  change pushed it over and `scripts/check_file_size.py` is a ratchet with no
  new entries allowed. The read-only cards moved to a new
  `web/src/agent-settings-panels.ts` (417 + 230 lines). A split, not a rewrite:
  the functions are unchanged apart from `export`.

DIFFICULTIES. Two test-environment traps, both fixed in the test rather than
worked around: the codex quota reader would otherwise read the developer's real
`~/.codex` (pinned with `codex_home=tmp_path`), and the opencode health probe
reaches for a local server that respx refuses to leave unrouted (answered with
the `ConnectError` a box without one gives).

EVIDENCE. `ruff check . && mypy .` clean; `python -m pytest` 989 passed, 1
skipped (the codex case of the hides-codex test, which skips because codex is
the backend that DOES read a quota); `npm run ci` 261 passed. All four `cmd:`
greps in the DoD are green, including the two that were red on base. The
`manual:` cross-surface check is still pending.

REFLECTION. The plan's own note that Steps 2/3 were stale was right, and the
same re-read caught a third stale claim (the `agent-view.ts` comment). Reading
the named line before trusting a Step's description of it paid for itself
twice here.
