# Clear the round-2 findings from the router extraction

- PRIORITY: 42
- TAGS: refactor, v0.2.0, backend
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a maintainer, I want the five MINOR/NIT findings review round 2 approved
over to be folded in, so that the router-extraction shape the next two lane-D
tasks copy does not carry a fail-open sweep, an untested guard, or stale
numbers.

## Steps

- [ ] R2.1 `scufris/api/routes.py` - drop `WebSocketRoute` from the
      `isinstance` tuple (line 67), from the `starlette.routing` import (line
      24), and from the comment above the tuple (lines 63-66), leaving `Mount`
      and a comment that names only the mounted sub-application. Confirmed:
      `issubclass(WebSocketRoute, Route)` is `False` and a `WebSocketRoute`
      appended to a `FastAPI()` is today swept up silently, so the branch
      re-opens for websockets the fail-open hole R1.3 closed for every other
      node kind - `BaseHTTPMiddleware` never sees a websocket scope, so a
      websocket endpoint added later would bypass `enforce_auth` while
      `iter_routes` dropped it from the boundary sweep. The app registers no
      websocket route (`rg -n 'WebSocketRoute|add_websocket_route|@app.websocket'
      scufris/` hits only `routes.py`), so nothing regresses and the first one
      added raises instead.
- [ ] R2.2 `tests/test_domain_routers.py` - add three `iter_routes` cases
      beside the existing router rigs, importing `iter_routes` from
      `scufris.api.routes`:
      (a) a `FastAPI()` with a `WebSocketRoute` appended to `.routes` raises
      `TypeError` - this one is RED on the base branch and is R2.1's proof;
      (b) a `FastAPI()` with a bare `starlette.routing.BaseRoute` subclass
      appended raises `TypeError` (`routes.py:69`);
      (c) a `FastAPI()` including a router with `prefix="/x"` raises
      `ValueError` (`routes.py:54`).
      (b) and (c) are characterization tests - the guards already fire, they
      were just never falsified. Prove each by deleting its `raise` in the
      working tree, watching the case go red, and restoring; record both
      mutation results in the work log.
- [ ] R2.3 `tasks/20260801-100425/DECISION.md:162,165` - restate `2924` as
      `2923` and `22 tests` as `25`, matching `TASK.md:181,271` where R1.9
      already corrected them. `tests/test_domain_routers.py` collects 25 tests
      today. Do NOT touch the same strings in that task's `REVIEW.md` or
      `RETRO.md`: those quote the stale values as the finding, and rewriting
      them would falsify the record.
- [ ] R2.4 `tests/test_domain_routers.py:365-370` - `FakeSessions`'s docstring
      claims "Real expiry semantics", but `prune` (line 405) appends to
      `pruned_at` and returns 0 without reading `idle`/`absolute`, and `get`
      (line 388) enforces both windows yet never renews `last_seen`. Narrow the
      docstring to what the fake honours - expiry enforced on read - and say
      `prune` only records that it was called.
- [ ] R2.5 `tests/test_domain_routers.py:418-424` - drop `self.settings`, and
      make `gate` and `throttle` locals in `AuthRig.__init__`. Confirmed unread:
      the only `auth_rig.*` reads in the file are `.app` (459) and `.sessions`
      (571, 578, 583, 599, 600, 604, 879); `rig.settings` (491) and `rig.gate`
      (485) belong to the other rig and stay.

## Definition of Done

- A websocket route is refused by the sweep rather than skipped
  (test: the `WebSocketRoute` case added in R2.2a, red on the base branch).
- Both pre-existing `iter_routes` guards are falsified
  (test: the R2.2b/R2.2c cases, each shown red against a working tree with its
  own `raise` deleted; results recorded in the work log).
- No websocket branch survives in the sweep
  (cmd: `rg -n 'WebSocketRoute' scufris/`, expected no match).
- No stale count survives in the decision record
  (cmd: `rg -n '2924|22 tests' tasks/20260801-100425/DECISION.md`, expected no
  match; two matches on the base branch).
- No drift (cmd: `python -m pytest -p no:randomly`).

## Notes

Source: `tasks/20260801-100425/REVIEW.md` round 2, findings R2.1-R2.5. The
verdict was APPROVE - none of these block - and the landing gate for
20260801-100425 chose to land without them, so they are carried here rather
than dropped.

R2.2 is the same lesson that task's own Reflection draws: a guard is not a
guard until it has been falsified. The round-1 fix for a fail-open sweep was
itself shipped without a red-first test.

Plan-time correction: the inherited DoD proof
`rg -n '2924|22 tests' tasks/20260801-100425/` cannot go green. `REVIEW.md:183,
249-250` and `RETRO.md:57` quote the stale values as the finding text, and
editing a review or retro record to satisfy a grep would be falsification. The
proof is scoped to `DECISION.md`, which is the only file R2.3 names.

Everything here is one file-local edit or one test addition; no interface,
storage or callers move, so there is nothing load-bearing to record in a
DECISION.md.
