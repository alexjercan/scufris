# Extract domain services and routers from application assembly

- STATUS: OPEN
- PRIORITY: 70
- TAGS: refactor, v0.2.0, backend, maintainability

## Story

As a maintainer, I want route assembly separated from project, agent,
diagnostics, chat, Telegram, and host-operation domain behavior, so that new
surfaces reuse one implementation and backend-specific correctness does not
depend on keeping a 2,000-line application factory synchronized by hand.

## Steps

- [ ] Characterize existing public routes, dependency construction, lifespan,
      stores, background services, and test override points with integration
      tests before moving code.
- [ ] Identify domain services that remove demonstrated duplication, especially
      orchestrator diagnostics, agent lifecycle/run control, projects, task
      artifacts, host inspection/actions/approvals/schedules, and Telegram
      operations.
- [ ] Extract one transport-independent orchestrator-turn service used by the
      landing chat, Telegram, and wake bridge, with typed inputs/results and no
      FastAPI or Telegram rendering concerns.
- [ ] Extract an agent-run service that owns launch, resume, cancel, status,
      completion, outcomes, and supervisor interaction for every caller.
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
- Landing chat, Telegram, and the wake bridge launch through the same
  orchestrator-turn service
  (test: `test_orchestrator_transports_share_turn_service`).
- Host proposal, approval, schedule, and audit routes delegate to explicit host
  services rather than remaining embedded in application assembly
  (test: `test_host_routes_delegate_to_domain_services`).
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
- Do not invent the future conversation schema here. The required seam is one
  transport-independent orchestrator service that 20260729-220835 can place a
  durable conversation around later.

## Flow State

- FLOW STEP: PLANNING
