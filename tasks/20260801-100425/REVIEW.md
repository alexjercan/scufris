# Review: Characterize app routes and extract the auth and host routers

- TASK: 20260801-100425
- BRANCH: refactor/extract-auth-host-routers

## Round 1

- REVIEWER: out-of-context (lanes: behavior/proofs; correctness/security/concurrency; design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (BLOCKER) tasks/20260801-100425/TASK.md:248 - the Evidence row
  records `python -m pytest -p no:randomly` as "866 passed, 1 skipped"; the
  branch actually collects 1016 tests and reports 1015 passed, 1 skipped (exit
  0, re-run by the primary), and base `master` collects 990, so no commit on
  this branch produced 867. Replace the row with the numbers an actual run
  gives.
  Response: fixed. The whole Evidence table is re-derived after the round-1
  fixes and now records `exit 0, 1018 passed, 1 skipped` (the three new auth
  cases from R1.2 raise the collected count from 1016 to 1019), plus the exit
  codes for
  `check_file_size.py`, `mypy`, `ruff` and `npm run ci`. The stale row is called
  out under Deviations: it was carried over from an earlier run instead of
  re-derived after the last commit.
- [x] R1.2 (MAJOR) tests/test_domain_routers.py:1 - Step 8 says build EACH
  router over fakes and the DoD says "the auth and host routers can be tested
  with fake services", but the rig builds only `build_host_router` and
  `build_hostconfig_router`; `rg -n 'build_auth_router|SessionGate|auth_middleware' tests/`
  returns no constructor, so the auth half of that DoD clause is unproven. Add
  a case building `build_auth_router(fake_gate, LoginThrottle(...))` on a bare
  `FastAPI()` over a fake `SessionStore`/`Settings`, driving
  `/api/auth/login|logout|session`, and include its paths in the isolation
  sweep.
  Response: fixed. `tests/test_domain_routers.py` gains `FakeSessions` (a
  `SessionStore` in a dict, with real idle/absolute expiry semantics) and
  `AuthRig`, which builds `build_auth_router(SessionGate(settings, fake_store),
  LoginThrottle(max_failures=3, window_seconds=60))` on a bare `FastAPI()`. The
  gate is real - it is the piece under test; the store beneath it is the fake.
  Three new cases: the login round trip (probe, wrong password mints nothing,
  success mints one record whose id and csrf ARE the cookies, prune swept, id
  rotates on re-login, logout revokes server-side), the cross-origin refusal
  landing before the throttle can burn, and the lockout with its `Retry-After`.
  `test_domain_router_dependency_isolation` now takes `auth_client` and drives
  `/api/auth/session|login|logout` under the same four traps. Settings come from
  `test_auth._settings`, the auth-domain-local helper the other auth suites
  already share, rather than a fourth copy of the hash.
  Falsified before trusting it: deleting `await gate.revoke(request)` from the
  logout route turns BOTH the new case and the isolation case red.
- [x] R1.3 (MINOR) scufris/api/routes.py:47 - `iter_routes` fails OPEN: a
  non-`Route` node without `original_router` is silently skipped, so if FastAPI
  renames `original_router`/`include_context` all four sweeps go back to
  covering less while still passing (the floors `checked_unauth > 40`,
  `checked_csrf > 10`, `checked >= 4` all still hold with the entire host, auth
  and hostconfig surface dropped). Raise on an unrecognized node unless it is a
  `Mount`/`WebSocketRoute`, and raise the floors to just under today's real
  counts.
  Response: fixed, both halves. `iter_routes` now raises `TypeError` on any node
  that is not a `Route`, an included router, a `Mount` or a `WebSocketRoute`;
  the two skipped kinds are named in a comment so the skip is a decision rather
  than a fallthrough. Measured today's real coverage - 81 unauthenticated
  route/methods, 31 CSRF, 5 mutating host routes - and moved the floors to
  `> 75`, `> 28` and `>= 5`.
- [x] R1.4 (MINOR) scufris/api/hostconfig.py:76 - the `try` now wraps the whole
  `ConfigChangeService.start`, where base wrapped only `config_builder.resolve`
  (422) and the explicit `building_for` check (409), so a `ConfigChangeRefused`
  or `ChangeInFlight` raised later (from `store.put`, say) now returns 422/409
  where base returned 500. Same widening at scufris/api/host.py:195, where
  `record_proposal` is now inside the `HostdError` mapping. Narrow each `try`
  to the call the base wrapped, or record the widening as an intentional
  deviation.
  Response: recorded as an intentional deviation (DECISION.md 4), not narrowed.
  Narrowing means splitting the service call back into the pieces the route used
  to hold, which is the design Step 5 removed. Checked that it changes no
  reachable status first - the raise-site census: `CannotPropose` has one site
  (the ACTIVATE refusal base mapped); `ConfigChangeRefused` is raised only in
  `hostconfig/resolve.py`, which is `_builder.resolve`, the call base wrapped
  (`changes.py:290` catches its own); `ChangeInFlight` has one site, the
  in-flight check; and `record_proposal` cannot raise `HostdError` because it
  only writes the store and fires hooks, and `_fire` swallows every hook
  exception (`host_approvals.py:228`). So the wider span is a difference in the
  `try`, not in any answer the app gives today. What it would cost is stated in
  DECISION.md.
- [x] R1.5 (MINOR) scufris/README.md:84 - the trust-boundary row still says the
  deny-by-default middleware lives "in `app.py`" and that
  `tests/test_auth_boundary.py` enumerates `app.routes`; both are now false and
  the second contradicts the new section 7 rule. Name `api/auth.py` and
  `iter_routes(app)`.
  Response: fixed. The row now names `api/auth.py::auth_middleware` and
  `iter_routes(app)`.
- [x] R1.6 (MINOR) scufris/README.md:272 - section 6 opens "`auth/` plus one
  middleware in `app.py`"; the middleware is now
  `api/auth.py::auth_middleware`. Repoint it and link section 7.
  Response: fixed. Section 6 now reads "`auth/` (the primitives) plus one
  middleware, `api/auth.py::auth_middleware`, which is also where the session
  gate and the `/api/auth/*` routes live - see section 7 for the router
  boundary".
- [x] R1.7 (MINOR) scufris/api/auth.py:223 - the newly written
  `auth_middleware` docstring says the boundary test "enumerates `app.routes`",
  the exact idiom `api/routes.py` and DECISION.md 4 forbid. Change to
  `iter_routes(app)`.
  Response: fixed. It now says the boundary test sweeps `iter_routes(app)`.
- [x] R1.8 (MINOR) scufris/auth/policy.py:34 - the `PUBLIC_PATHS` comment (34)
  and the `OPERATOR_ONLY_PATTERN` comment (73) both still promise the guards
  enumerate `app.routes`; the doc sweep missed them. Update both to
  `iter_routes`.
  Response: fixed, both. Every surviving `app.routes` mention in the repo now
  describes the OLD idiom on purpose and says so - `scufris/README.md:337`,
  `api/routes.py:6`, `app.py:2854` and the two test docstrings that contrast
  the two. No comment claims a guard still uses it.
- [x] R1.9 (MINOR) tasks/20260801-100425/TASK.md:182 - the Close-out table's
  "host route bodies in `create_app`: 27" does not match base: `git show
  master:scufris/app.py` holds 21 host, stats, processes and config route
  decorators. Restate as 21.
  Response: fixed. Restated as 21, and the `app.py` line count in the same
  table corrected to 2923 (it moved by one with the round-1 edits).
- [x] R1.10 (MINOR) scufris/hostconfig/__init__.py:66 - `ConfigChangeService`,
  `NoRunningBuild` and `Proposer` are re-exported here but no caller uses the
  path (`app.py` and `api/hostconfig.py` both import from
  `..hostconfig.service`), while `api/hostconfig.py` imports
  `ConfigChange`/`ConfigChangeRefused` from `..hostconfig` and
  `ChangeInFlight`/`UnknownChange` from `..hostconfig.service` - three
  spellings for one package. Pick the package facade and import all six from
  `..hostconfig`, or delete lines 65-68 and 82-86.
  Response: fixed by picking the facade. `api/hostconfig.py` imports all six
  from `..hostconfig`, `app.py` takes `ConfigChangeService` from there too, and
  `tests/test_domain_routers.py` follows. `rg 'hostconfig.service'` now returns
  nothing outside the package. The package docstring says the rule explicitly
  (a caller outside the package imports from `__init__` and nowhere else) and
  its module table gains the `service` row it was missing.
- [x] R1.11 (MINOR) scufris/api/__init__.py:1 - the 20-line module docstring
  restates the `api/` boundary rules from README section 7 and DECISION.md 1-2,
  making three copies of one normative statement (global AGENTS.md: explanatory
  prose belongs in the record). Cut to two lines plus a pointer to README
  section 7.
  Response: fixed. Cut to one sentence plus a pointer to README section 7.
- [x] R1.12 (MINOR) scufris/api/auth.py:157 - `SessionGate.deny` reads no
  instance state, only `request`; it is a free function on a class whose job is
  session identity. Make it a module-level `def deny(request, status, detail)`.
  Response: fixed. `deny` is a module-level function in `api/auth.py`; the
  seven call sites in `auth_middleware` call it directly.
- [x] R1.13 (NIT) scufris/api/host.py:211 - the propose route awaits
  `gate.requester_identity` (a session-store read) as an argument to
  `approvals.propose`, so it now runs BEFORE the service's ACTIVATE refusal,
  where master refused first and never touched the store; one extra store
  transaction on a path that always 422s. Resolve the identity lazily, or
  accept and note it.
  Response: accepted and noted (DECISION.md 4). Resolving it lazily means
  handing the service a callable instead of a `Requester`, which is a worse
  interface than one store read on an always-422 path is worth.
- [x] R1.14 (NIT) scufris/api/auth.py:144 - `SessionGate.prune` calls
  `auth_now()` afresh, where the master login passed the same `moment` used for
  the throttle decision; sub-millisecond, but a real deviation from the
  byte-for-byte claim. Take `now` as a parameter from the login route.
  Response: fixed. `SessionGate.prune(now)` takes the instant; the login route
  passes the same `moment` the throttle decided on.
- [x] R1.15 (NIT) scufris/host/overview.py:54 - `fresh()` has no caller outside
  the class. Rename to `_fresh` so the module's public surface is the one call
  the Step asked for.
  Response: fixed. Renamed to `_fresh`; both call sites are inside the class.
- [x] R1.16 (NIT) scufris/api/sse.py:19 - the `SseEvent` docstring repeats the
  module docstring's "shared by agent turns and by host applies" sentence word
  for word. Drop it from the class.
  Response: fixed. The class docstring is one line now.
- [x] R1.17 (NIT) scufris/api/host.py:330 - `cancel_host_action` and
  `api/hostconfig.py:121 cancel_config_change` take a `request: Request` they
  never read; carried over from master, free to delete now that the bodies
  moved.
  Response: fixed. Both parameters are gone. The route contract test is
  unchanged and still green - a handler parameter that is not a path or query
  parameter does not appear in the route table.
- [x] R1.18 (NIT) scufris/api/routes.py:9 - the module docstring enumerates its
  four call sites by filename, a census that goes stale the moment a fifth
  guard appears. State the rule, not the list.
  Response: fixed. The list is replaced by the rule: anything deriving the
  app's real surface must walk routers too, or it covers less while still
  passing.

Verification by the primary, independent of the lanes:

- `python -m pytest -p no:randomly` -> exit 0, 1016 collected, 1015 passed, 1
  skipped. This is the re-derivation behind R1.1.
- `cd web && npm run ci` -> exit 0 (the lanes reported this proof skipped).
- `mypy .` -> Success, no issues in 205 source files.
- `wc -l scufris/app.py` -> 2924, under the 3000 proof; base 3785.
- `python scripts/check_file_size.py` -> exit 0, and
  `git diff master...HEAD -- scripts/check_file_size.py` is empty, so no
  `ALLOWLIST` growth.
- `rg -n 'api/' scufris/README.md` -> exit 0.
- `rg -n 'build_auth_router|SessionGate|auth_middleware' tests/` -> one prose
  mention, no constructor. This is the re-derivation behind R1.2.
- Base host route decorator count -> 21, behind R1.9.
- Middleware order preserved: `enforce_auth` registered before `log_requests`,
  so the logger stays outermost, matching master.
- Step 1 is honoured: `5bb67e4` adds only `tests/test_route_contract.py`, with
  no `app.py` edit, and the expected route-table literal is untouched by later
  commits.
- The ACTIVATE refusal is not forked: `mcp_host_tools/actions.py` proposes over
  `POST /api/host/actions`.

Process signal: the DoD proof line named `npm run test:e2e`, a script that
never existed in `web/package.json`; it was corrected to `npm run ci` mid-task
and disclosed. The plan gate accepted an unverified command.

Process signal: three mypy errors introduced by this task's own Step 1 were
fixed in Step 8's commit, so the characterization test shipped red under
`mypy` for three commits.

Process signal: both DoD tests were falsified before being trusted, and the
falsifications are described concretely in the record. That is the behavior to
keep.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

Every round-1 finding is verified against the code. R1.1, R1.2, R1.3, R1.5-R1.12
and R1.14-R1.18 are confirmed fixed; R1.4 and R1.13 are disputed-and-accepted,
and their reasoning was re-derived independently (the `CannotPropose`,
`ConfigChangeRefused` and `ChangeInFlight` raise-site census holds, and
`record_proposal` cannot raise `HostdError` because `_fire` swallows every hook
exception). R1.9 is confirmed for TASK.md only; the same numbers went stale in
DECISION.md, which is R2.3. No regression came out of the fix commit: the two
tests it touched were strengthened (floors raised), not weakened, and
`tests/test_route_contract.py` is byte-identical to `5bb67e4`.

No open BLOCKER or MAJOR, so the verdict is APPROVE. The five findings below
are MINOR/NIT and do not block; they are worth folding in before landing.

- [ ] R2.1 (MINOR) scufris/api/routes.py:67 - `WebSocketRoute` is pre-skipped,
  but the app registers no websocket route (`rg -n 'websocket' scufris/` hits
  only `api/routes.py` itself), so the branch is speculative AND it re-opens
  the hole the fix commit closed for every other node kind: Starlette's
  `BaseHTTPMiddleware` never sees a websocket scope, so a websocket endpoint
  added later would bypass `enforce_auth` and `iter_routes` would silently drop
  it from the boundary sweep instead of raising. Delete `WebSocketRoute` from
  the `isinstance` tuple, from the `starlette.routing` import and from the
  comment, so the first websocket route makes `iter_routes` raise and forces
  the gating decision. `Mount` stays - the static dist is real today.
- [ ] R2.2 (MINOR) scufris/api/routes.py:69 - the new fail-closed
  `raise TypeError` (and the pre-existing prefix/tags `ValueError` at line 54)
  have no test: `rg -n 'iter_routes' tests/` returns only sweep call sites, so
  the guard R1.3 asked for shipped unfalsified - the exact failure mode the
  record's own Reflection names. Add a case appending a bare
  `starlette.routing.BaseRoute` subclass to a `FastAPI().routes` and asserting
  `TypeError`, plus one that includes a router with `prefix="/x"` and asserts
  `ValueError`.
- [ ] R2.3 (MINOR) tasks/20260801-100425/DECISION.md:162 - two numbers
  contradict the rig and the values R1.9 corrected in TASK.md: Consequences
  says `3785 -> 2924 lines` where `wc -l scufris/app.py` is 2923 (TASK.md:181
  says 2923), and line 165 says `tests/test_domain_routers.py`, 22 tests` where
  the file collects 25 (TASK.md:271 says 25). Restate as 2923 and 25.
- [ ] R2.4 (NIT) tests/test_domain_routers.py:365 - `FakeSessions`'s docstring
  claims "Real expiry semantics", but `prune` (line 405) ignores `idle` and
  `absolute` and removes nothing, and `get` never renews `last_seen` the way
  the real store does. Narrow the sentence to what the fake honours (expiry on
  read) and say `prune` only records that it was called.
- [ ] R2.5 (NIT) tests/test_domain_routers.py:419 - `AuthRig.settings`, `.gate`
  and `.throttle` are never read by any test (only `.app` and `.sessions` are).
  Drop `self.settings` and make `gate` and `throttle` locals in `__init__`.

Verification by the primary, independent of the out-of-context lane:

- `python -m pytest -p no:randomly` -> exit 0, 1018 passed, 1 skipped. This is
  the re-derivation behind the R1.1 tick: the number TASK.md records is the
  number the rig produces on this commit.
- `mypy .` -> Success, no issues in 205 source files.
- `ruff check .` -> All checks passed.
- `python scripts/check_file_size.py` -> exit 0.
- `rg -n 'api/' scufris/README.md` -> exit 0.
- `cd web && npm run ci` -> exit 0, webpack compiled successfully.
- `wc -l scufris/app.py` -> 2923, under the 3000 proof. This is the
  re-derivation behind R2.3.
- `rg -n 'iter_routes' tests/` -> sweep call sites and docstrings only, no
  guard assertion. This is the re-derivation behind R2.2.
- Proofs 1-7 all pass on their stated criteria. There are no `manual:` proofs
  on this task, so nothing is pending a user check.

- The out-of-context lane could not measure the sweep floors with
  `SCUFRIS_WEB_DIST` present; counts can only rise with a dist mounted, so the
  75/28/5 floors stay safe either way.

Process signal: R2.2 is the same lesson the record's Reflection already draws -
a guard is not a guard until it has been falsified - recurring on the fix that
introduced the guard. The round-1 fix for a fail-open sweep was itself shipped
without a red-first test.
