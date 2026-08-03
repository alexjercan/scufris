# Decision: Router factories over a frozen deps dataclass

- DATE: 20260803-021500
- STATUS: ACCEPTED
- TASK: 20260801-100425
- TAGS: refactor,backend,host,http,testability

## Context

`create_app` held 79 route decorators, two middlewares and the entire service
graph as local closure state across 2900 lines. Two consequences drove this task:

- **nothing could be tested without the whole app.** Exercising
  `/api/host/actions/{id}/approve` meant a state directory, a SQLite database, an
  `AgentStore`, a `ProjectStore`, an environment-derived settings object, a
  supervisor and a lifespan - all to prove that one route calls one method and
  maps one exception onto one status;
- **routes read names bound below them.** `/api/host/digests` read `scheduler`
  and `digests` constructed 1200 lines later, and `_manual_runs` was declared
  AFTER the route body that added to it. Both worked only because Python resolves
  a closure at call time and no request arrives before `create_app` returns. That
  is a latent ordering bug wearing a working program as a disguise.

The extraction had to pick how a router receives its dependencies, where the
domain rules the routes were holding should go, and what to do with the one route
whose taxonomy does not fit.

## Decision

**1. `build_*_router(Deps(...))` - a factory over a frozen deps dataclass, not
FastAPI `Depends`.**

The dependency graph is per-APP, not per-request. `HostApprovalService`,
`HostScheduler` and `ConfigChangeService` are constructed once and live for the
process; `Depends` exists to resolve something per REQUEST. Using it for a
process-lifetime singleton means writing a provider that returns a global -
which is the global this task set out to remove.

A missing dependency becomes a construction error at the `include_router` line
rather than a `NameError` on the first live request. That is precisely the bug
class above, closed rather than relocated.

`SessionGate` is an ordinary field for the same reason: the middleware, the host
routes and `/api/agents/{id}/chat` must share ONE gate, and a per-request
provider would let their answers to "who is asking" drift.

**2. A route translates; the service decides.**

Four rules moved out of route bodies, each leaving behind only a status mapping:

| Rule | Now in | Route maps |
|---|---|---|
| `activate` may not be proposed directly | `HostApprovalService.propose` | `CannotPropose` -> 422 |
| what a schedule is; keeping a fire-and-forget run alive | `HostScheduler.start_now` | `ValueError` -> 422 |
| one build per repository; resolve, mint, start, cancel | `ConfigChangeService` | `ConfigChangeRefused` -> 422, `ChangeInFlight` -> 409, `NoRunningBuild` -> 409 |
| the helper's refusal codes | `api/errors.py` | one shared table |

The MCP host toolset (`mcp_host_tools/actions.py`) proposes by calling
`POST /api/host/actions` over the API rather than holding a second copy - checked,
not assumed - so there is one implementation of the `activate` refusal.

The configuration flow reaches `hostd.propose(ACTIVATE, ...)` directly and then
`approvals.record_proposal`, bypassing `approvals.propose`. Not an exception: the
rule is about a CALLER naming a store path, and there the path is one this server
built from a revision it resolved itself. `create_app._propose_activation`
carries that reasoning at the call site.

**3. `/api/config` rides with the host router until the settings router lands.**

It is app configuration and carries the `app` tag, not `host`. It is in
`api/host.py` anyway because it is ONE route; a module holding a single four-line
handler exists to satisfy a taxonomy. Its real home is the settings router in
**20260729-103712**, alongside `/api/agent/config` and the `settings_store`
surface, and moving it twice is worse than once. It also reads
`MIN_HOST_OVERVIEW_TTL` off the host overview cache, so its current neighbours
are not arbitrary.

**4. A service call is wrapped as ONE unit, so the `except` around it is wider
than the base's.**

Base wrapped exactly the call that could raise: `config_builder.resolve` in the
`ConfigChangeRefused` handler, `hostd.propose` in the `HostdError` handler, with
`record_proposal` and the in-flight check outside. A route that delegates to
`ConfigChangeService.start` or `HostApprovalService.propose` cannot do that - the
service IS the unit, which is decision 2 - so the mapping now covers everything
those methods do.

Accepted, after checking that it changes no reachable status. The raise-site
census:

| Error | Raised at | Reachable from the wider span? |
|---|---|---|
| `CannotPropose` | `host_approvals.py` (the ACTIVATE refusal) | no - one site, the one base mapped |
| `ConfigChangeRefused` | `hostconfig/resolve.py` only; `changes.py` catches its own | no - only from `resolve`, the call base wrapped |
| `ChangeInFlight` | `service.start`'s explicit check | no - one site |
| `HostdError`/`HostdUnavailable` | the socket client | no - `record_proposal` only writes the store and fires hooks, and `_fire` swallows every hook exception |

So the widening is a theoretical difference in the span, not a behavioural one
today. It is recorded rather than engineered around because narrowing it means
splitting the service call back into the pieces the route used to hold, which is
the design this task exists to remove. What it costs: if a store write inside
`record_proposal` ever starts raising `HostdError`, the propose route answers 422
or 503 where base answered 500. Review round 1, R1.4.

The same shape covers the propose route's requester lookup: `requester_identity`
is now resolved as an ARGUMENT to `approvals.propose`, so it runs before the
service's ACTIVATE refusal, where base refused first and never read the session
store. One extra store read on a path that always 422s, accepted for the same
reason - resolving it lazily means handing the service a callable instead of a
`Requester`, which is a worse interface than the cost is worth. Review round 1,
R1.13.

**5. `iter_routes`, never `for route in app.routes`.**

FastAPI 0.139 changed `include_router`: it appends one opaque node that resolves
its routes lazily, so the plain idiom stops seeing a route the moment it moves
onto a router - silently. Three of the four users of that idiom are GUARDS (the
OpenAPI tag assignment, the auth-boundary sweep, the operator-only coverage
sweep) and each would have kept passing while covering less.
`api/routes.py::iter_routes` refuses to descend into a router included with a
`prefix` or `tags`, because callers read `route.path` and `route.tags` off what
comes back and under an include-time prefix those are no longer what the served
route answers to.

It fails CLOSED in the same spirit: a node that is neither a `Route`, an included
router, a `Mount` nor a `WebSocketRoute` raises rather than being skipped. The
first version skipped it, which would have rebuilt the exact bug - if FastAPI
renames `original_router`, every sweep silently covers less and still reports
green. The four sweeps also hold their floors just under today's real counts (81
unauthenticated route/methods, 31 CSRF, 5 mutating host routes) instead of at
token values that stayed satisfied with the whole router surface dropped.

## Alternatives considered

**FastAPI `Depends` with `app.dependency_overrides` in tests.** Rejected.
`dependency_overrides` is a mutable dict on the app: a test that forgets to clear
it leaks into the next. Passing fakes into a constructor has no teardown. It also
inverts the failure mode - an unsatisfiable dependency surfaces per request
instead of at assembly.

**A `Protocol` per dependency instead of concrete types on `HostDeps`.**
Rejected as speculative: there is one production implementation of each, and
`tests/test_domain_routers.py` passes fakes through `cast(Any, ...)`, the
established convention in this repo's test suite. A protocol earns its place when
a second real implementation appears.

**One `api/host.py` for both routers.** The combined body is over the 600-line
source cap that `scripts/check_file_size.py` enforces, and the two are genuinely
separable: one talks to the root helper over a socket, the other to the build
registry.

**Rehoming `/api/config` now.** See decision 3.

**Leaving the domain rules in the routes and only moving the code.** That would
have produced smaller modules with the same problem - a route that decides is a
route the Telegram surface and the MCP toolset can disagree with.

## Consequences

**Good.**

- `scufris/app.py`: 3785 -> 2924 lines, and every new module is under the
  600-line cap without an `ALLOWLIST` entry.
- Every host route is now drivable on a bare `FastAPI()` over fakes, with no
  database, no stores and no lifespan (`tests/test_domain_routers.py`, 22 tests).
- The late-binding hazard is gone by construction, not by discipline.
- `20260801-100441` and `20260729-103712` have a shape to follow rather than a
  judgement call to re-make.

**Costs.**

- A router cannot be included twice with different dependencies without being
  built twice. Nothing wants that today.
- `HostDeps` has eleven fields, which reads as a lot until you notice it is the
  honest count of what the host surface depends on - previously spread across
  1200 lines of closure scope.
- `create_app` now has an ORDER constraint that is load-bearing: the scheduler
  and digest store must be constructed before the host router is included. It
  fails loudly if violated, which is the improvement, but it is a real constraint
  a later edit can hit.

**Watch for.**

- A new route added to `api/host.py` that reaches for a global instead of adding
  a `HostDeps` field. `test_domain_router_dependency_isolation` traps `Settings`,
  `AgentStore`, `ProjectStore` and `Database` by patching `__init__` on the
  classes - patching the defining module's attribute does NOT work, because
  `from ..config import Settings` binds at import time (measured: the first
  version of that test passed against a router that called `Settings()`).
- Any future router included with a `prefix` or `tags` will make `iter_routes`
  raise, which is deliberate: that is exactly when its four callers need
  revisiting.
