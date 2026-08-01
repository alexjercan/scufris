# Delegate legacy /api/agent/* routes to orchestrator diagnostics

- STATUS: OPEN
- PRIORITY: 74
- TAGS: bug,v0.2.0,agents,backend
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102148

## Story

As an operator using the landing page, I want the legacy `/api/agent/*`
endpoints to answer from the same diagnostics service as the scoped routes, so
that switching the orchestrator away from Codex stops leaking stale Codex
account, usage, and memory data into the UI.

## Steps

- [ ] Add failing tests asserting `/api/agent/info`, `/api/agent/account`,
      `/api/agent/usage`, `/api/agent/memory`, `/api/agent/health`,
      `/api/agent/tools`, and `/api/agent/mcp` match their scoped equivalents
      for every backend.
- [ ] Make each legacy route an adapter over the diagnostics service. The
      unconditional `read_usage(resolve_codex_home(settings))` calls in
      `scufris/app.py` (~lines 3545-3571) are the specific leak to remove.
- [ ] Keep the legacy response schemas and OpenAPI paths stable; only the
      source of the data changes.
- [ ] Grep for remaining direct Codex-account reads outside the service and
      remove or justify each one.
- [ ] Verify the landing and agent-settings UIs render the unsupported and
      empty states without misleading labels or stale values; adjust the
      frontend where a placeholder currently implies real data.

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
