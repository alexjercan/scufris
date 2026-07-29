# Extract domain services and routers from application assembly

- STATUS: OPEN
- PRIORITY: 75
- TAGS: refactor,v0.1.0,backend,maintainability

## Story

As a maintainer, I want route assembly separated from project, agent,
diagnostics, chat, and Telegram domain behavior, so that new surfaces reuse one
implementation and backend-specific correctness does not depend on keeping a
2,000-line application factory synchronized by hand.

## Steps

- [ ] Characterize existing public routes, dependency construction, lifespan,
      stores, background services, and test override points with integration
      tests before moving code.
- [ ] Identify domain services that remove demonstrated duplication, especially
      orchestrator diagnostics, agent lifecycle, chat/run control, projects,
      task artifacts, and Telegram operations.
- [ ] Extract FastAPI routers that receive typed services/dependencies while
      retaining a small application factory for configuration and assembly.
- [ ] Move models and helpers only when their ownership becomes clearer; avoid
      a mechanical one-file-to-many-files split with the same coupling.
- [ ] Remove compatibility-route duplication by delegating to the extracted
      services and keep OpenAPI paths and response behavior stable.
- [ ] Preserve shutdown, background task, callback, test injection, static
      frontend, request logging, and exception behavior.
- [ ] Add route-contract and application-start integration tests, then update
      architecture documentation and file ownership guidance.

## Definition of Done

- Existing API and browser integration suites pass without public route drift
  (cmd: `python -m pytest && cd web && npm run ci && npm run test:e2e`).
- Legacy and scoped orchestrator routes share the same diagnostics service
  (test: `test_legacy_agent_routes_delegate_to_scoped_diagnostics`).
- `create_app` is limited to dependency/lifespan/router/static assembly
  (test: `test_application_factory_assembles_domain_routers`).
- Domain routers can be tested with explicit fake services rather than global
  settings or unrelated store construction
  (test: `test_domain_router_dependency_isolation`).

## Notes

- Epic: 20260729-102145.
- Depends on: 20260729-102147 and 20260729-102148.
- Relevant code: `scufris/app.py`.
- Refactor against characterized contracts. Do not combine this task with new
  product behavior.

## Flow State

- FLOW STEP: PLANNING
