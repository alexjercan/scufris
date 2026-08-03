# Characterize app routes and extract the auth and host routers

- PRIORITY: 72
- TAGS: refactor, v0.2.0, backend, host, maintainability
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100419

## Story

As a maintainer, I want the public route surface characterized by tests and
the auth and host routes moved out of the application factory, so that the
remaining extraction proceeds against a contract instead of against hope.

## Steps

- [x] Pin the surface FIRST, in `tests/test_route_contract.py`, and commit it
      before `scufris/app.py` is touched: the full sorted route table of
      `create_app()` (path, methods, response-model name, `include_in_schema`,
      assigned OpenAPI tag) against a literal expected table; the `app.state`
      keys other code reads; the lifespan's background services; and the five
      `create_app` override points. This test is a CHARACTERIZATION test - it
      is green on base by construction and stays green after.
- [x] Add the `scufris/api/` package and move the SSE plumbing into
      `scufris/api/sse.py` verbatim: the `_SseEvent` protocol, `_last_event_id`
      and `_relay_bus_sse` (it closes over nothing but its arguments).
      `scufris/app.py` imports them; the agent routes keep using them.
- [x] Extract authentication into `scufris/api/auth.py`: a `SessionGate` over
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
- [x] Move `_HostOverviewCache` to `scufris/host/overview.py` as
      `HostOverviewCache`, taking the `MIN_HOST_OVERVIEW_TTL` floor and the
      single-flight `asyncio.Lock` with it, so the route is one call instead of
      a cache protocol spread across the factory. Update the two tests in
      `tests/test_app.py` that import it from `scufris.app`.
- [x] Give the host routes services to delegate to, one per verb family:
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
- [x] Extract the host surface into two routers, both built by a factory over a
      frozen deps dataclass rather than over `create_app`'s scope:
      `scufris/api/host.py` (`/api/stats`, `/api/processes`,
      `/api/host/overview`, `/api/host/actions*`, `/api/host/digests*`,
      `/api/host/audit`, `/api/config`) and `scufris/api/hostconfig.py`
      (`/api/host/config/changes*`). The request/response models move with
      their router; `hostd_http_error` goes to `scufris/api/errors.py`. Two
      modules, not one, because the combined body is over the 600-line source
      cap.
- [x] Reorder `create_app` so the host router's dependencies exist before it is
      included: `digests`, `scheduler_store`, `_run_scheduled_checks` and
      `scheduler` are built at lines 2741-2864 today and reached only by late
      closure binding from routes at lines 1525-1573. Move their construction
      above the include and grep the moved bodies for any other name bound
      after their current definition point.
- [x] Add `tests/test_domain_routers.py`: build each router over fakes on a
      bare `FastAPI()` and drive it with `TestClient`, asserting the routes
      delegate to the services and that no database, `AgentStore`,
      `ProjectStore` or env-derived global settings object is constructed.
- [x] Record the decision in `tasks/20260801-100425/DECISION.md`: router
      factories over a frozen deps dataclass instead of FastAPI `Depends`
      overrides, and why `/api/config` rides with the host router until the
      settings router lands in 20260729-103712.
- [x] Record the router and service ownership boundaries in `scufris/README.md`
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
  (cmd: `python -m pytest -p no:randomly && cd web && npm run ci`).

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

## Close-out

### What and why

`create_app` held 79 route decorators and the whole service graph as local
closure state. The host surface is now three modules under `scufris/api/` -
`host.py`, `hostconfig.py` and the shared `errors.py` - each built by a factory
over a frozen deps dataclass, and `create_app` assembles them.

The measured shape:

| | base (`9f6c40c`) | now |
|---|---|---|
| `scufris/app.py` | 3785 | 2923 |
| host route bodies in `create_app` | 21 | 0 |
| new modules over the 600-line cap | - | 0 (largest is `api/host.py` at 413) |

Four rules left the routes and became domain code, each with the route reduced
to a status mapping: the `activate` refusal (`HostApprovalService.propose`), the
unknown-schedule refusal and the manual-run task set (`HostScheduler.start_now`),
and the whole configuration-build flow (`ConfigChangeService`).

### Alternatives

- **FastAPI `Depends` plus `app.dependency_overrides` in tests.** Rejected;
  DECISION.md 1. The dependency graph is per-app, not per-request, and overrides
  are global mutable state a test can forget to clear.
- **One `api/host.py` holding both routers.** The combined body is over the
  600-line source cap, and the two are separable - one talks to the root helper,
  the other to the build registry.
- **Rehoming `/api/config` immediately.** Rejected; DECISION.md 2. It is one
  route whose real home is the settings router in 20260729-103712, and moving it
  twice is worse than once.

### Difficulties and diagnosis

**The isolation test passed against a deliberately broken router.** The first
version of `test_domain_router_dependency_isolation` patched
`scufris.config.Settings`. It was falsified by adding a real `Settings()` call
inside `get_stats` - and the test still passed, because
`from ..config import Settings` binds the class into `scufris.api.host` at import
time and patching the defining module's attribute leaves that binding alone. The
trap now patches `__init__` on the classes themselves (`Settings`, `AgentStore`,
`ProjectStore`, `Database`), which catches every import spelling. Both DoD tests
were falsified this way before being trusted; the second falsification (re-inline
the `activate` refusal in the route) correctly turned
`test_activate_is_refused_by_the_service_not_the_route` red.

**Late closure binding.** `/api/host/digests` read `scheduler` and `digests`
constructed 1200 lines below it, and `_manual_runs` was declared AFTER the route
body that added to it. Both only worked because no request arrives before
`create_app` returns. The scheduler and digest store now move above the include;
the task set moved into `HostScheduler`. The `HostDeps` construction is the point
where a missing dependency fails, so the class of bug is closed rather than
relocated.

**`ConfigChangeService.bus` was typed `object | None`,** which `relay_bus_sse`
cannot take. Tightened to `EventBus[ConfigBuildEvent] | None`.

### Deviations from the plan

- **`npm run test:e2e` does not exist.** The DoD named it; `web/package.json` has
  no such script and there is no browser-driver suite in the repo. The DoD line
  is corrected to `npm run ci`, which runs format, lint, the jsdom suite and the
  build. Nothing was skipped - the proof as written named a command that has
  never existed.
- **Three pre-existing mypy errors in `tests/test_route_contract.py`** (landed by
  this task's own Step 1) are fixed here: `sorted(route.methods)` over
  `set[str] | None`, `list(route.tags)` over `list[str | Enum]`, and
  `HostOverview(nixos_generations=[])` naming a field that does not exist.
- **A service call is wrapped as one unit, so two `except` spans are wider than
  base's**, and the propose route resolves the requester before the service's
  ACTIVATE refusal. Both accepted with a raise-site census showing no reachable
  status change; DECISION.md 4.

### Round 1 fixes

Every finding except R1.4 and R1.13 was fixed; those two are recorded as
intentional deviations in DECISION.md 4 rather than engineered around. The two
that changed behaviour rather than prose:

- **The auth half of the DoD was unproven** (R1.2). `tests/test_domain_routers.py`
  built only the two host routers, so "the auth and host routers can be tested
  with fake services" was half a claim. It now builds `build_auth_router` over a
  real `SessionGate` and an in-memory `FakeSessions`, driving the login round
  trip, session rotation, server-side revocation, the origin check that runs
  before the throttle, and the lockout - and the isolation sweep drives the auth
  paths under the same booby traps. Falsified: deleting the `gate.revoke` from
  the logout route turns both the new test AND the isolation test red.
- **`iter_routes` failed open** (R1.3). An unrecognized node was skipped, so a
  FastAPI rename of `original_router` would have put every sweep back to covering
  less while still passing - the exact bug the function was written to close. It
  now raises on anything that is not a `Route`, an included router, a `Mount` or
  a `WebSocketRoute`, and the four sweep floors moved from token values (40, 10,
  4) to just under the real counts (75, 28, 5).

### Evidence

Measured after round 1's fixes, on this branch:

| Proof | Result |
|---|---|
| `tests/test_route_contract.py` (4 tests) | green, unchanged - the extraction introduced no drift |
| `tests/test_domain_routers.py` (25 tests) | green; both DoD tests falsified first, and the auth half falsified again in round 1 |
| `wc -l scufris/app.py` | 2923 (proof: under 3000) |
| `python scripts/check_file_size.py` | exit 0; `git diff master...HEAD -- scripts/check_file_size.py` empty, so no `ALLOWLIST` growth |
| `rg -n 'api/' scufris/README.md` | exit 0 |
| `python -m pytest -p no:randomly` | exit 0, 1018 passed, 1 skipped |
| `mypy .` | Success, no issues in 205 source files |
| `ruff check .` / `ruff format .` | clean |
| `cd web && npm run ci` | exit 0 |

The row this table replaced claimed "866 passed, 1 skipped", a number no commit
on this branch ever produced (round 1, R1.1). It was carried over from an earlier
run rather than re-derived after the last commit; the counts above come from a
run made after every round-1 fix.

### Reflection

Writing the characterization test FIRST (Step 1, committed before `app.py` was
touched) is what made the rest boring: every one of the four seams that could
have silently drifted - a path, a response model, an `app.state` key, an OpenAPI
tag - was pinned before anything moved, and the route table stayed byte-identical
through three commits of surgery.

The thing worth carrying forward is that a guard is not a guard until it has been
falsified. `iter_routes` exists because `for route in app.routes` would have kept
passing while covering less, and the isolation test's first trap would have done
exactly the same thing. Both were caught by asking "what would make this red?"
rather than by reading the code.

### Next

`20260801-100441` and `20260729-103712` extract the agent/project and settings
surfaces. They follow this shape: a `build_*_router` over a frozen deps
dataclass, the domain rule in the service, and a `tests/test_domain_routers.py`
case per router. `/api/config` moves to the settings router when it lands.
