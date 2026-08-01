# Align Telegram and the UI with orchestrator diagnostics

- STATUS: OPEN
- PRIORITY: 73
- TAGS: bug,v0.2.0,telegram,backend,frontend
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100415

## Story

As an operator reading Telegram, I want `/settings` and `/stats` to report the
same effective backend, model, health, and quota semantics as the web UI, so
that the two surfaces never disagree about which orchestrator is running.

## Steps

- [ ] Add failing tests comparing Telegram `/settings` and `/stats` output
      against the diagnostics service for Codex, Claude, OpenCode, and mock.
- [ ] Point the Telegram providers in `scufris/telegram/` at the diagnostics
      service, removing the backend branching they carry today.
- [ ] Render unsupported capabilities as an explicit statement rather than a
      zero, a blank, or a Codex-shaped quota line, in
      `scufris/telegram/render.py`.
- [ ] Reconcile the wording between Telegram and the web UI so the same state
      reads the same way on both.
- [ ] Extend the cross-backend diagnostics contract documentation with the
      Telegram surface and the rule that new surfaces consume the service.

## Definition of Done

- Telegram matches the web diagnostics for every backend
  (test: `test_telegram_settings_match_orchestrator_diagnostics`).
- Telegram never prints Codex quota or rollout counts for a non-Codex
  orchestrator (test: `test_telegram_hides_codex_account_data_for_other_backends`).
- No Telegram module reads a backend-specific account source directly
  (cmd: `rg -n "resolve_codex_home|read_usage" scufris/telegram/`, expected empty).
- The contract documents every consuming surface
  (cmd: `rg -n "Telegram|landing|agent settings" scufris/README.md`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- Backend and account information feels consistent across the landing page,
  agent settings, and Telegram (manual: user check).

## Notes

- Epic: 20260729-102145.
- Depends on the legacy-route delegation task, so all three surfaces converge
  on one service in order rather than at once.
- This task carries the epic's backend-consistency manual acceptance, since it
  is the last surface to join.
