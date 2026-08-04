# Clear the round-2 findings from the router extraction

- PRIORITY: 42
- TAGS: refactor, v0.2.0, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145

## Story

As a maintainer, I want the five MINOR/NIT findings review round 2 approved
over to be folded in, so that the router-extraction shape the next two lane-D
tasks copy does not carry a fail-open sweep, an untested guard, or stale
numbers.

## Steps

- [x] R2.1 `scufris/api/routes.py` - drop `WebSocketRoute` from the
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
- [x] R2.2 `tests/test_route_iteration.py` (NEW - see close-out; the planned
      home, `tests/test_domain_routers.py`, would have breached the 900-line test
      cap) - add three `iter_routes` cases, importing `iter_routes` from
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
- [x] R2.3 `tasks/20260801-100425/DECISION.md:162,165` - restate `2924` as
      `2923` and `22 tests` as `25`, matching `TASK.md:181,271` where R1.9
      already corrected them. `tests/test_domain_routers.py` collects 25 tests
      today. Do NOT touch the same strings in that task's `REVIEW.md` or
      `RETRO.md`: those quote the stale values as the finding, and rewriting
      them would falsify the record.
- [x] R2.4 `tests/test_domain_routers.py:365-370` - `FakeSessions`'s docstring
      claims "Real expiry semantics", but `prune` (line 405) appends to
      `pruned_at` and returns 0 without reading `idle`/`absolute`, and `get`
      (line 388) enforces both windows yet never renews `last_seen`. Narrow the
      docstring to what the fake honours - expiry enforced on read - and say
      `prune` only records that it was called.
- [x] R2.5 `tests/test_domain_routers.py:418-424` - drop `self.settings`, and
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

## Close-out

### What and why

The five round-2 findings from `tasks/20260801-100425/REVIEW.md`, folded in
before the next two lane-D tasks copy the router-extraction shape.

`iter_routes` now refuses a `WebSocketRoute` instead of skipping it (R2.1). The
skip was the fail-open hole R1.3 closed for every other node kind, still open
for the one node kind no HTTP middleware can cover: `BaseHTTPMiddleware` never
sees a websocket scope, so a websocket endpoint added later would have bypassed
`enforce_auth` while the boundary sweep silently dropped it. The app registers
no websocket route today, so nothing regresses; the first one added raises.

The two pre-existing guards are now falsified rather than assumed (R2.2), the
sibling decision record carries the corrected numbers (R2.3), and the auth test
rig no longer overstates its fake or hold attributes nobody reads (R2.4, R2.5).

### Deviation: R2.2 landed in a new file

The plan put the three cases in `tests/test_domain_routers.py`. They pushed it
to 933 lines against a 900-line test cap, and `tests/test_check_file_size.py`
says in as many words: split the file, do not add an allowlist entry. They went
to a new `tests/test_route_iteration.py` instead.

This is the better home regardless of the cap. The three cases are about
`iter_routes`'s fail-closed contract, not about the domain routers, and
`test_route_contract.py` - the other candidate - pins `create_app`'s public
surface, a different subject again. `test_domain_routers.py` ends at 894 lines
- one over the 893 it started at, since R2.4 and R2.5 both touched it - with
room under the cap that the previous shape did not leave.

### Alternatives considered

- Allowlisting `test_domain_routers.py`: refused by the check's own message.
- Folding the cases into `test_route_contract.py`: that file is a
  characterization pin on `create_app`, green by construction; unit assertions
  on a helper's refusals do not belong to that claim.
- Leaving `WebSocketRoute` in the tuple and covering websockets with a separate
  guard: speculative, and it keeps a fail-open branch alive to do it. YAGNI.

### Difficulties and diagnosis

Only the file-size cap, diagnosed straight from the failing check's output
(`tests/test_domain_routers.py: 933 lines, cap 900`) after the full suite went
red on `test_check_file_size.py`. No investigation needed beyond reading it.

### Evidence

| Proof | Result |
|-|-|
| R2.2a websocket case | RED on base (`DID NOT RAISE TypeError`), green after R2.1 |
| R2.2b mutation: `raise TypeError` -> `continue` | `test_an_unrecognized_route_node_is_refused` RED (and the websocket case with it) |
| R2.2c mutation: `raise ValueError` -> `pass` | `test_a_router_included_with_a_prefix_is_refused` RED |
| `rg -n 'WebSocketRoute' scufris/` | no match (exit 1) |
| `rg -n '2924\|22 tests' tasks/20260801-100425/DECISION.md` | no match (exit 1); two matches on base |
| `python -m pytest -p no:randomly` | green, exit 0 |
| `ruff check .`, `ruff format --check .`, `mypy .` | clean; 229 source files |
| `nix flake check` | all 7 checks passed |
| `tests/test_domain_routers.py` collected | 25 on base, 894 lines after the split |

Both `raise`s were restored from a pre-mutation copy and the restore confirmed
by `git diff --stat` before the final suite run.

### Reflection

The task's own Notes named the lesson - a guard is not a guard until it has
been falsified - and R2.2 paid it off for both round-1 guards. Worth noting the
round-2 finding that produced R2.1 was itself only visible because R1.3's fix
had no red-first test: the `WebSocketRoute` in the skip tuple looked like a
deliberate exemption until someone asked what covers a websocket instead.

The file-size cap did the job a cap is for: it refused a plan that would have
put three helper-contract tests inside an 900-line suite about something else,
and the forced split produced the better arrangement. Cheaper to obey than to
argue with.
