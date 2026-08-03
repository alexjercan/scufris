# Characterize app routes and extract the auth and host routers

- PRIORITY: 72
- TAGS: refactor, v0.2.0, backend, host, maintainability
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100419

## Story

As a maintainer, I want the public route surface characterized by tests and
the auth and host routes moved out of the application factory, so that the
remaining extraction proceeds against a contract instead of against hope.

## Steps

- [ ] Pin the surface FIRST, in `tests/test_route_contract.py`, and commit it
      before `scufris/app.py` is touched: the full sorted route table of
      `create_app()` (path, methods, response-model name, `include_in_schema`,
      assigned OpenAPI tag) against a literal expected table; the `app.state`
      keys other code reads; the lifespan's background services; and the five
      `create_app` override points. This test is a CHARACTERIZATION test - it
      is green on base by construction and stays green after.
- [ ] Add the `scufris/api/` package and move the SSE plumbing into
      `scufris/api/sse.py` verbatim: the `_SseEvent` protocol, `_last_event_id`
      and `_relay_bus_sse` (it closes over nothing but its arguments).
      `scufris/app.py` imports them; the agent routes keep using them.
- [ ] Extract authentication into `scufris/api/auth.py`: a `SessionGate` over
      (`Settings`, `SessionStore`) owning `session_of`, `issue`, `revoke`,
      `deny`, `operator_identity`, `requester_identity` and `caller_is_agent`;
      an `auth_middleware(gate, ...)` factory returning the `enforce_auth`
      dispatch; and `build_auth_router(gate, throttle)` carrying
      `/api/auth/login|logout|session` plus `LoginRequest` and `AuthSession`.
      `create_app` keeps `validate_auth_config`, the token mint and the store
      construction, and registers `enforce_auth` BEFORE `log_requests` exactly
      as now (Starlette applies middleware in reverse, so the logger must stay
      outermost). `caller_is_agent` still serves `/api/agents/{id}/chat` in
      `app.py` (line 3094), so the gate is shared, not auth-router-private.
- [ ] Move `_HostOverviewCache` to `scufris/host/overview.py` as
      `HostOverviewCache`, taking the `MIN_HOST_OVERVIEW_TTL` floor and the
      single-flight `asyncio.Lock` with it, so the route is one call instead of
      a cache protocol spread across the factory. Update the two tests in
      `tests/test_app.py` that import it from `scufris.app`.
- [ ] Give the host routes services to delegate to, one per verb family:
      - `HostApprovalService.propose(kind, args, requester)` in
        `scufris/host_approvals.py` - owns the `ActionKind.ACTIVATE` refusal
        (a domain rule about what may be proposed, not a route concern), the
        `hostd.propose` call and `record_proposal`. Confirm what the MCP host
        toolset's propose path (`scufris/mcp_host_tools/actions.py`) does today
        and either route it through this or leave it untouched; do not fork the
        rule into two copies.
      - `HostScheduler.start_now(schedule)` in `scufris/scheduler.py` - owns
        the unknown-schedule refusal and the `_manual_runs` task set that
        currently keeps the fire-and-forget run alive from the route body.
      - `ConfigChangeService` in `scufris/hostconfig/service.py` over
        (store, builder, supervisor, proposer, settings) - owns resolve, the
        one-build-per-repository refusal, the record write, the supervised
        stream start, and cancel, raising domain errors the router maps.
- [ ] Extract the host surface into two routers, both built by a factory over a
      frozen deps dataclass rather than over `create_app`'s scope:
      `scufris/api/host.py` (`/api/stats`, `/api/processes`,
      `/api/host/overview`, `/api/host/actions*`, `/api/host/digests*`,
      `/api/host/audit`, `/api/config`) and `scufris/api/hostconfig.py`
      (`/api/host/config/changes*`). The request/response models move with
      their router; `hostd_http_error` goes to `scufris/api/errors.py`. Two
      modules, not one, because the combined body is over the 600-line source
      cap.
- [ ] Reorder `create_app` so the host router's dependencies exist before it is
      included: `digests`, `scheduler_store`, `_run_scheduled_checks` and
      `scheduler` are built at lines 2741-2864 today and reached only by late
      closure binding from routes at lines 1525-1573. Move their construction
      above the include and grep the moved bodies for any other name bound
      after their current definition point.
- [ ] Add `tests/test_domain_routers.py`: build each router over fakes on a
      bare `FastAPI()` and drive it with `TestClient`, asserting the routes
      delegate to the services and that no database, `AgentStore`,
      `ProjectStore` or env-derived global settings object is constructed.
- [ ] Record the decision in `tasks/20260801-100425/DECISION.md`: router
      factories over a frozen deps dataclass instead of FastAPI `Depends`
      overrides, and why `/api/config` rides with the host router until the
      settings router lands in 20260729-103712.
- [ ] Record the router and service ownership boundaries in `scufris/README.md`
      (section 7, the HTTP surface, and section 8, the module map) so
      20260801-100441 and 20260729-103712 follow the same shape.

## Definition of Done

- The public route table, app state, lifespan and override points are asserted
  and unchanged (test: `tests/test_route_contract.py`, all four tests). This is
  the extraction's gate, not a red proof: it is green on base by construction,
  and any drift introduced by the extraction turns it red.
- Host proposal, approval, schedule, and audit routes delegate to explicit host
  services (test:
  `tests/test_domain_routers.py::test_host_routes_delegate_to_domain_services`).
- The auth and host routers can be tested with fake services, without global
  settings or unrelated store construction (test:
  `tests/test_domain_routers.py::test_domain_router_dependency_isolation`).
- `scufris/app.py` shrinks by the extracted surface
  (cmd: `wc -l scufris/app.py`, expected under 3000; base is 3785).
- Every new module is under the source cap and no allowlist entry is added
  (cmd: `python scripts/check_file_size.py`, plus
  `git diff scripts/check_file_size.py` showing no `ALLOWLIST` growth).
- The boundaries are written down (cmd: `rg -n 'api/' scufris/README.md`).
- Existing API and browser suites pass without drift
  (cmd: `python -m pytest -p no:randomly && cd web && npm run ci && npm run test:e2e`).

## Notes

Base measurements on `master` at `9f6c40c`:

- `scufris/app.py` is 3785 lines; `create_app` spans 891-3731 and holds 79
  route decorators, two middlewares, and the whole service graph as local
  closure state.
- `wc -l scufris/app.py` -> 3785 (red against the under-3000 proof).
- `python -m pytest tests/test_route_contract.py` -> exit 4, no such file (red).
- `python -m pytest tests/test_domain_routers.py` -> exit 4, no such file (red).
- `rg -n 'api/' scufris/README.md` -> exit 1 (red).
- `python scripts/check_file_size.py` -> exit 0. This one is a GUARD, not a red
  proof: it passes today and must still pass with the new modules, whose whole
  point is to be under the 600-line source cap. `ALLOWLIST` holds
  `scufris/app.py` and `tests/test_app.py`; app.py stays over the cap after
  this task, so its entry stays.

Discovered facts that shape the extraction:

- Late closure binding is the ordering hazard. `/api/host/digests` (1525) and
  `/api/host/digests/run` (1539) read `scheduler` and `digests`, constructed at
  2741 and 2851. The host SSE routes (1586, 1836) call `_relay_bus_sse`,
  defined at 3033. A router factory binds its dependencies at construction, so
  these must move above the include.
- `_manual_runs` is declared at 1572, AFTER the route body that adds to it -
  the same pattern, inside one function. It moves into `HostScheduler`.
- The auth gate is not host-only: `_caller_is_agent` is used by
  `/api/agents/{agent_id}/chat` (3094), and `_session_of` is used by the
  middleware, the auth routes, and every host identity helper.
- `app.state` is a real contract. Tests and the Telegram wiring read
  `supervisor`, `agents`, `projects`, `db`, `auth_required`, `api_token`,
  `sessions`, `hostd`, `host_actions`, `host_supervisor`, `host_approvals`,
  `config_changes`, `config_supervisor`, `digests`, `host_scheduler`,
  `host_checks_task`, `telegram_bot`, `telegram_task`,
  `telegram_approval_ops`. The route-contract test pins them.
- OpenAPI tags are assigned by path in `_route_tags` after every route exists
  (3717-3720), so moving a route onto a router does not change its tag as long
  as its path is unchanged. The contract test asserts the tag anyway.
- `tests/test_app.py` imports `_HostOverviewCache` and `MIN_HOST_OVERVIEW_TTL`
  from `scufris.app` (lines 333, 372); those imports change with the move.

Assumptions:

- `/api/config` goes in the host router because this task's Steps place it
  there, even though it is app configuration and carries the `app` tag. It is
  one route with no other home until the settings router lands; recorded in
  DECISION.md rather than silently rehomed.
- The suite is verified with `-p no:randomly`. `pytest-randomly` exposes a
  known order-dependent failure in
  `tests/test_app.py::test_orchestrator_chat_uses_server_cwd`, owned by
  20260803-043935. A refactor of this size must not be judged against a flake
  it did not cause.
- Behavior is preserved exactly: no status code, response model, path, tag, log
  line, or exception mapping changes. Anything that looks like a bug in the
  moved code is recorded, not fixed here.
- Refactor against characterized contracts. Do not combine with new product
  behavior.
