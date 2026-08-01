# Characterize app routes and extract the auth and host routers

- STATUS: OPEN
- PRIORITY: 72
- TAGS: refactor,v0.2.0,backend,host,maintainability
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100419

## Story

As a maintainer, I want the public route surface characterized by tests and
the auth and host routes moved out of the application factory, so that the
remaining extraction proceeds against a contract instead of against hope.

## Steps

- [ ] Characterize the current surface first: an integration test asserting the
      full public route table (path, method, response model) of the app built
      by `create_app`, plus its lifespan, background services, and test
      override points.
- [ ] Extract the auth routes (`/api/auth/login`, `/logout`, `/session`) into a
      router that receives typed dependencies rather than closure state.
- [ ] Extract the host routes (`/api/host/*`, `/api/stats`, `/api/processes`,
      `/api/config`) into a host router backed by explicit host services for
      inspection, actions, approvals, schedules, and config changes.
- [ ] Keep `_HostOverviewCache` and the SSE event plumbing working; move them
      with the router they serve rather than leaving them in assembly.
- [ ] Preserve OpenAPI paths, response models, request logging, and exception
      behavior exactly; the characterization test is the gate.
- [ ] Record the router and service ownership boundaries in `scufris/README.md`
      so the successor tasks follow the same shape.

## Definition of Done

- The public route table is asserted and unchanged
  (test: `test_public_route_contract_is_stable`).
- Host proposal, approval, schedule, and audit routes delegate to explicit host
  services (test: `test_host_routes_delegate_to_domain_services`).
- The auth and host routers can be tested with fake services, without global
  settings or unrelated store construction
  (test: `test_domain_router_dependency_isolation`).
- `scufris/app.py` shrinks by the extracted surface
  (cmd: `wc -l scufris/app.py`, expected well under the pre-task 3745).
- Existing API and browser suites pass without drift
  (cmd: `python -m pytest && cd web && npm run ci && npm run test:e2e`).

## Notes

- Epic: 20260729-102145.
- Depends on the Telegram diagnostics task, so the backend-truth work is done
  before routes move and does not have to be rediscovered mid-refactor.
- Relevant code: `scufris/app.py` (3745 lines; `create_app` spans ~924-3684).
- Refactor against characterized contracts. Do not combine with new product
  behavior.
