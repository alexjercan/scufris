# Notes: Clear the round-2 findings from the router extraction

## What changes

No runtime behavior on any request path. What changes is what the route sweep
does with a node it was never given: today a `WebSocketRoute` is silently
skipped, after this it raises, so the first websocket route added to the app
forces an explicit gating decision instead of quietly falling outside
`enforce_auth`. The two `iter_routes` guards also gain the tests that make them
guards rather than assertions.

## Surfaces

| File | Why |
|-|-|
| `scufris/api/routes.py:24,67,64-68` | R2.1: drop `WebSocketRoute` from the import, the `isinstance` tuple and the comment |
| `tests/test_domain_routers.py` | R2.2: two new `iter_routes` cases (TypeError, ValueError); R2.4/R2.5 fake cleanup |
| `tests/test_domain_routers.py:365` (`FakeSessions` docstring) | R2.4: claims "Real expiry semantics" it does not have |
| `tests/test_domain_routers.py:419` (`AuthRig.__init__`) | R2.5: three never-read attributes |
| `tasks/20260801-100425/DECISION.md:162,165` | R2.3: `2924` -> `2923`, `22 tests` -> `25` |

## Data and interfaces

No production signature changes.

```python
def iter_routes(target: Any) -> Iterator[Route]   # unchanged; the Mount/WebSocketRoute
                                                  # tuple narrows to (Mount,)
```

`AuthRig.__init__` narrows its surface:

```python
class AuthRig:
    sessions: FakeSessions
    app: FastAPI
    # settings dropped; gate and throttle become locals
```

`FakeSessions` keeps its methods; only the docstring narrows to what it honours
(expiry on read; `prune` records the call and removes nothing; `get` does not
renew `last_seen`).

## Sketches

R2.1 (illustrative):

```python
-from starlette.routing import Mount, Route, WebSocketRoute
+from starlette.routing import Mount, Route
...
-        # A mounted sub-application (the static dist) and a websocket route have
-        # no HTTP route table to contribute ...
-        if isinstance(route, (Mount, WebSocketRoute)):
+        # A mounted sub-application (the static dist) has no HTTP route table to
+        # contribute, and every caller asks about HTTP routes. Skipped
+        # deliberately, and named, so the skip is a decision, not a fallthrough.
+        if isinstance(route, Mount):
            continue
        raise TypeError(...)
```

R2.2 (illustrative):

```python
def test_iter_routes_refuses_an_unknown_node() -> None:
    app = FastAPI()
    app.routes.append(_BareRoute())          # a bare starlette BaseRoute subclass
    with pytest.raises(TypeError):
        list(iter_routes(app))

def test_iter_routes_refuses_a_prefixed_router() -> None:
    app = FastAPI()
    app.include_router(APIRouter(), prefix="/x")
    with pytest.raises(ValueError):
        list(iter_routes(app))
```

## Shape

```
iter_routes(app)
   |
   +-- Route / APIRoute .......... yield
   +-- _IncludedRouter ........... prefix or tags? -> ValueError   [R2.2 untested]
   |                                else recurse
   +-- Mount ..................... skip (static dist is real today)
   +-- WebSocketRoute ............ skip            <-- R2.1 DELETE this arm
   +-- anything else ............. TypeError       [R2.2 untested]

Why R2.1 matters: enforce_auth is BaseHTTPMiddleware, which never sees a
websocket scope. A websocket route added later would be ungated AND invisible
to the boundary sweep that is supposed to catch exactly that - the same
fail-open hole R1.3 closed for every other node kind.
```

## Consequences and open questions

- All five pointers verified against the tree today: `routes.py:67` is the
  `isinstance` tuple, `:69` the `raise TypeError`, `DECISION.md:162,165` carry
  `2924` and `22 tests`, `test_domain_routers.py:365`/`:419` are the docstring
  and `AuthRig.__init__`. No drift to re-derive.
- R2.1 makes adding a websocket route a hard failure until someone decides how
  it is authenticated. That is the intent, and it is a real (small) cost paid by
  whoever adds the first one: they must edit `routes.py` as well.
- R2.2's `_BareRoute` needs a `starlette.routing.BaseRoute` subclass that is not
  a `Route` and has no `original_router`. A minimal stub in the test file is
  enough; do not reach for a real transport.
- R2.3 edits a `tasks/` record. `tasks/` is normally append-only history - this
  is an explicit correction of a number the same review already corrected in
  TASK.md, so it is in scope, but the DoD grep
  (`rg -n '2924|22 tests' tasks/20260801-100425/`) will still match
  `REVIEW.md:183,249-250`, which quote the wrong values as the finding itself.
  **Open question for planning: the DoD command as written cannot pass.** Narrow
  it to `DECISION.md`, or scope it to lines outside the review quotes.
- R2.4/R2.5 are pure test hygiene; the existing `test_domain_routers.py` run is
  their whole proof.
