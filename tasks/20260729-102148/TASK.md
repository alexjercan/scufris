# Unify backend-aware orchestrator and Telegram diagnostics

- STATUS: OPEN
- PRIORITY: 75
- TAGS: bug, v0.2.0, agents, backend, telegram
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As an operator, I want every Scufris surface to describe the orchestrator's
effective backend and capabilities truthfully, so that switching between
Codex, Claude, OpenCode, and mock never leaks stale Codex-specific account,
usage, health, memory, or MCP information.

## Steps

- [ ] Add parametrized failing tests that configure each supported backend and
      compare landing, scoped-agent, and Telegram settings results.
- [ ] Extract one backend-aware orchestrator diagnostics service for info,
      account, usage, memory, health, and visible tools.
- [ ] Make legacy `/api/agent/*`, `/api/agents/orchestrator/*`, and Telegram
      providers delegate to the same service.
- [ ] Return explicit empty or unsupported states where a backend has no
      Codex-equivalent quota, memory, or MCP capability.
- [ ] Remove duplicated backend branching and static model reads that can drift
      from the persisted orchestrator record.
- [ ] Verify the landing and settings UIs render unsupported data without
      misleading labels or stale values.
- [ ] Document the cross-backend diagnostics contract for future harnesses.

## Definition of Done

- Codex, Claude, OpenCode, and mock produce consistent backend/model/account
  data across all orchestrator surfaces
  (test: `test_orchestrator_surfaces_are_backend_consistent`).
- Non-Codex orchestrators never expose Codex quota or rollout counts
  (test: `test_non_codex_orchestrator_hides_codex_account_data`).
- Telegram `/settings` and `/stats` use the same effective diagnostics as the
  web UI (test: `test_telegram_settings_match_orchestrator_diagnostics`).
- No legacy route independently constructs Codex-only account data
  (test: `test_legacy_agent_routes_delegate_to_scoped_diagnostics`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on: 20260729-102147.
- Must land before: 20260729-103712. The refactor extracts the shared service
  established here instead of rediscovering backend-aware diagnostics while
  moving routes.
- The scoped per-agent endpoints are the closest existing source of truth.
- Keep compatibility routes, but make them adapters rather than independent
  implementations.
