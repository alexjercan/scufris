# Review: Clear the round-2 findings from the router extraction

- TASK: 20260803-061210
- BRANCH: refactor/router-round2

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [x] R1.1 (NIT) tasks/20260803-061210/TASK.md:129 - the close-out says
  `test_domain_routers.py` "returns to 895 lines" and the Evidence table
  repeats "895 lines after the split". `wc -l` gives 894, and master's copy is
  893, so the file is +1 against the base rather than a return to a lower
  count. Restate as 894 and drop "returns to"; the cap headroom claim stands
  either way.
  - Response: corrected to 894 in both places, with the +1 against the base
    stated rather than implied. Re-verified by `wc -l` against
    `git show master:tests/test_domain_routers.py`.
- [ ] R1.2 (NIT) tests/test_route_iteration.py:32 - the websocket case builds a
  raw `starlette.routing.WebSocketRoute` and appends it to `app.routes`, but
  the way a websocket actually enters this app is `@app.websocket("/ws")`,
  which produces `APIWebSocketRoute`. Both land on the same branch so the pin
  is sound; using the decorator would pin the path a maintainer will really
  take. Take it or leave it.
  - Response: left. `APIWebSocketRoute` subclasses `WebSocketRoute`, so the
    decorator form reaches the same branch through the same `isinstance`
    check; swapping it in would pin one extra layer of FastAPI's construction
    without covering a case the raw node misses.

Both findings are NIT, so nothing open blocks the verdict.

Verified independently by the recording pass, not taken from the reviewer:

- `tests/test_route_iteration.py` against a copy of the branch with master's
  `scufris/api/routes.py` restored: `test_a_websocket_route_is_refused_rather_than_skipped`
  FAILS and the other two pass. That is proof 1 - R2.2a is red on the base and
  R2.1 is what turns it green - re-derived rather than accepted.
- `wc -l tests/test_domain_routers.py` 894 against `git show master:...` 893,
  which is where R1.1 comes from.
- `tests/test_domain_routers.py` collects 25, so the `22 tests` -> `25`
  correction in `tasks/20260801-100425/DECISION.md` is the true number.
- `rg -n 'WebSocketRoute' scufris/` exit 1 (proof 3);
  `rg -n '2924|22 tests' tasks/20260801-100425/DECISION.md` exit 1 (proof 4).
- `python -m pytest -p no:randomly` exit 0 (proof 5); `ruff check .` and
  `ruff format --check .` exit 0; `mypy .` clean over 229 source files;
  `tatr check` exit 0.

The out-of-context reviewer additionally reproduced proof 2, both mutations, in
a throwaway copy: `raise TypeError` -> `continue` reds the websocket and
`Exotic` cases, `raise ValueError` -> `pass` reds the prefix case. All three
guards falsify.

Observations, not findings. The `Mount` skip is now the only branch of
`iter_routes` with no direct test; it is exercised only when the web dist
exists, which it does not under pytest. Pre-existing, and outside this task's
claim. And `scufris/api/openapi.py:113` walks the routes during app
construction, so the first websocket added surfaces the new refusal as a
boot-time `TypeError` rather than a test failure - which is the intent the
Steps state, and the app registers no websocket today.

No `manual:` proofs on this task, so there are no pending user checks.

Inspection commands:

```bash
cd "$(sprout show refactor/router-round2)"
git diff master...HEAD
python -m pytest -p no:randomly
wc -l tests/test_domain_routers.py && git show master:tests/test_domain_routers.py | wc -l
```
