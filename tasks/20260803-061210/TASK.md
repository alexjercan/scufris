# Clear the round-2 findings from the router extraction

- PRIORITY: 42
- TAGS: refactor, v0.2.0, backend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a maintainer, I want the five MINOR/NIT findings review round 2 approved
over to be folded in, so that the router-extraction shape the next two lane-D
tasks copy does not carry a fail-open sweep, an untested guard, or stale
numbers.

## Steps

- [ ] R2.1 `scufris/api/routes.py:67` - delete `WebSocketRoute` from the
      `isinstance` tuple, the `starlette.routing` import and the comment. The
      app registers no websocket route, so the branch is speculative, and it
      re-opens for websockets exactly the fail-open hole R1.3 closed for every
      other node kind: `BaseHTTPMiddleware` never sees a websocket scope, so a
      websocket endpoint added later would bypass `enforce_auth` while
      `iter_routes` silently dropped it from the boundary sweep. Deleting it
      makes the first websocket route raise and forces the gating decision.
      `Mount` stays - the static dist is real today.
- [ ] R2.2 `scufris/api/routes.py:69` - the fail-closed `raise TypeError` and
      the pre-existing prefix/tags `ValueError` (line 54) have no test, so the
      guard R1.3 asked for shipped unfalsified. Add a case appending a bare
      `starlette.routing.BaseRoute` subclass to a `FastAPI().routes` and
      asserting `TypeError`, plus one including a router with `prefix="/x"` and
      asserting `ValueError`.
- [ ] R2.3 `tasks/20260801-100425/DECISION.md:162,165` - restate `2924` as
      `2923` and `22 tests` as `25`, matching the rig and the values R1.9
      already corrected in TASK.md.
- [ ] R2.4 `tests/test_domain_routers.py:365` - `FakeSessions`'s docstring
      claims "Real expiry semantics", but `prune` ignores `idle`/`absolute` and
      removes nothing, and `get` never renews `last_seen`. Narrow it to what the
      fake honours (expiry on read) and say `prune` only records the call.
- [ ] R2.5 `tests/test_domain_routers.py:419` - `AuthRig.settings`, `.gate` and
      `.throttle` are never read; only `.app` and `.sessions` are. Drop
      `self.settings`; make `gate` and `throttle` locals in `__init__`.

## Definition of Done

- The route sweep raises on a websocket route instead of skipping it, and both
  `iter_routes` guards are falsified by a test that goes red without them
  (test: the new `iter_routes` guard cases in `tests/test_domain_routers.py`).
- No stale count survives in the task records
  (cmd: `rg -n '2924|22 tests' tasks/20260801-100425/`, expected no match).
- No drift (cmd: `python -m pytest -p no:randomly`).

## Notes

Source: `tasks/20260801-100425/REVIEW.md` round 2, findings R2.1-R2.5. The
verdict was APPROVE - none of these block - and the landing gate for
20260801-100425 chose to land without them, so they are carried here rather
than dropped.

R2.2 is the same lesson that task's own Reflection draws: a guard is not a
guard until it has been falsified. The round-1 fix for a fail-open sweep was
itself shipped without a red-first test.
