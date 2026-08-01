# Extract the remaining routers and reduce create_app to assembly

- STATUS: OPEN
- PRIORITY: 70
- TAGS: refactor, v0.2.0, backend, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100441

## Story

As a maintainer, I want the remaining project, agent, chat, and diagnostics
routes in their own routers and `create_app` reduced to assembly, so that new
surfaces reuse one implementation instead of depending on a hand-synchronized
application factory.

## Steps

- [ ] Extract the project router (`/api/projects/*`, `/projects/{id}` SPA
      passthroughs) over an explicit project service.
- [ ] Extract the agent router (`/api/agents/*`, `/agents/{id}` passthroughs)
      delegating to the agent-run and diagnostics services.
- [ ] Extract the chat router (`/api/chat`, `/api/chat/stream`,
      `/api/chat/reset`) delegating to the orchestrator-turn service.
- [ ] Extract the legacy compatibility router (`/api/agent/*`) as thin
      adapters over the same services, keeping OpenAPI paths and responses
      stable.
- [ ] Reduce `create_app` to configuration, dependency construction, lifespan,
      router registration, and static mounting.
- [ ] Preserve static frontend serving, request logging, exception handlers,
      route tagging, and test injection points.
- [ ] Update `scufris/README.md` module map and file ownership guidance to the
      shipped layout.

## Definition of Done

- `create_app` is limited to dependency, lifespan, router, and static assembly
  (test: `test_application_factory_assembles_domain_routers`).
- The public route table is unchanged from the characterization baseline
  (test: `test_public_route_contract_is_stable`).
- Every domain router is testable with fake services
  (test: `test_domain_router_dependency_isolation`).
- Legacy and scoped orchestrator routes share the diagnostics service
  (test: `test_legacy_agent_routes_delegate_to_scoped_diagnostics`).
- `scufris/app.py` is under the repository file-size cap
  (cmd: `python -m pytest -k file_size` or the repo's size guard check).
- Existing API and browser suites pass without drift
  (cmd: `python -m pytest && cd web && npm run ci && npm run test:e2e`).

## Notes

- Epic: 20260729-102145.
- Depends on the turn/run service extraction; the routers delegate to services
  that already exist rather than creating them while moving code.
- The file-size guard from 20260731-171420 applies to the resulting modules.
- Refactor only. No new product behavior.
