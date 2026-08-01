# Extract the backend-aware orchestrator diagnostics service

- STATUS: OPEN
- PRIORITY: 75
- TAGS: bug, v0.2.0, agents, backend, telegram
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100413

## Story

As an operator, I want one backend-aware service that answers info, account,
usage, memory, health, and visible tools for any orchestrator, so that
Codex-specific readers stop being the implicit definition of what an agent can
report.

## Steps

- [ ] Add parametrized failing tests that configure Codex, Claude, OpenCode,
      and mock orchestrators and compare the scoped `/api/agents/{id}/*`
      diagnostics results across backends.
- [ ] Extract a transport-independent diagnostics service covering info,
      account, usage, memory, health, and visible tools, with typed results
      that can express supported, empty, and unsupported.
- [ ] Move the backend branching now spread across `scufris/app.py`
      (`_agent_is_codex`, `resolve_codex_home` reads at lines ~3120-3190) into
      the service, resolving capability from the backend rather than from a
      name comparison at each call site.
- [ ] Read the effective model and account from the persisted orchestrator
      record instead of static settings reads that can drift.
- [ ] Point the scoped `/api/agents/{id}/*` routes at the service; leave the
      legacy `/api/agent/*` routes and Telegram untouched in this task.
- [ ] Document the cross-backend diagnostics contract for future harnesses in
      `scufris/README.md`.

## Definition of Done

- All four backends produce consistent backend/model/account shapes on the
  scoped surface
  (test: `test_scoped_diagnostics_are_backend_consistent`).
- A non-Codex orchestrator never reports Codex quota or rollout counts
  (test: `test_non_codex_orchestrator_hides_codex_account_data`).
- Unsupported capabilities return an explicit unsupported state, not a silent
  empty or a stale value
  (test: `test_unsupported_diagnostics_are_explicit`).
- The service reads the persisted orchestrator record, not static settings
  (test: `test_diagnostics_follow_the_persisted_orchestrator_record`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the state migration lane, so the persisted orchestrator record it
  reads is already on the transactional store.
- The scoped per-agent endpoints are the closest existing source of truth;
  this task promotes them into a service rather than inventing a new contract.
- Legacy-route delegation and Telegram alignment are the two successor tasks.
