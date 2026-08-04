# Notes: Fix the examples the package carve broke

## What changes

Before: four of the thirteen `examples/*.py` die on their own. Measured on a
freshly `uv sync`ed tree, 2026-08-04:

| example | exit | cause |
|-|-|-|
| `host_agent.py` | 1 | `ModuleNotFoundError: No module named 'test_host_actions'` |
| `telegram_approval.py` | 1 | same import, then the loop-thread guard behind it |
| `comms_loop.py` | 1 | `RuntimeError: a transaction cannot be opened on a thread with a running event loop` |
| `telegram_bot.py` | 1 | same `RuntimeError` |

After: all thirteen exit 0, and the four join the opt-in list in
`tests/test_examples.py` so `nix flake check` fails when one rots again.

TASK.md's Story has one wrong premise, corrected here: `nix flake check` DOES
run examples. `flake.nix:262` runs `python -m pytest`, and
`tests/test_examples.py` runs each name in its `OFFLINE` tuple as a subprocess.
The four broken scripts simply are not in that tuple - the gate exists and has a
hole, rather than not existing. That changes Step 3 from "invent a check" to
"extend the check that is already there", which is much cheaper.

## Two independent defects, not one

**1. Stale test-tree path (the carve).** `6d998c8` moved
`tests/test_host_actions.py` to `packages/hostd/tests/`. `host_agent.py:43` and
`telegram_approval.py:39` still insert `ROOT / "tests"` and import
`host_files` / `host_runner` from it. The pytest suites survived because pytest
puts `packages/hostd/tests` on `sys.path` itself during collection.

Confirmed by probe: `PYTHONPATH=packages/hostd/tests python examples/host_agent.py`
exits 0 and prints the full round trip.

**2. The event-loop transaction guard.** `_refuse_the_event_loop_thread`
(`packages/core/src/scufris_core/engine.py:111`, landed in `3e71b44` per
20260801-100409 DECISION.md 1) raises when `Database.transaction()` is entered
on a thread with a running loop. `create_app` -> `state_database` ->
`upgrade_to_head` opens a transaction, so any example that calls `create_app`
from inside `asyncio.run(...)` now dies. Three do: `comms_loop.py:76`,
`telegram_bot.py:134`, `telegram_approval.py:125`.

`auth_session.py` and `host_agent.py` build the app on the sync side already,
which is why neither hits this. The guard is correct and stays; the examples are
what is wrong.

`telegram_approval.py` carries both defects, so fixing only the path leaves it
red - this is the trap in taking TASK.md's Steps literally.

## Surfaces

| file | why |
|-|-|
| `examples/host_agent.py:43` | repoint the fixture path at `packages/hostd/tests` |
| `examples/telegram_approval.py:39` | same repoint |
| `examples/telegram_approval.py:110-125,247-257` | hoist `Settings` + `create_app` out of `_run` into `main` |
| `examples/comms_loop.py:62-76,132-134` | same hoist |
| `examples/telegram_bot.py:118-134,207-209` | same hoist |
| `tests/test_examples.py:33-38` | add the four names to `OFFLINE` |
| `tasks/20260804-041340/TASK.md` | correct the Story's "runs no example" premise |

Nothing under `scufris/`, `packages/*/src/` or `flake.nix` changes. No new file.

## Data and interfaces

No public API changes. Three private example helpers change signature, each
gaining what its `main` now builds:

```python
# examples/comms_loop.py
async def run(app: FastAPI, proj: Path) -> int: ...        # was run() -> int

# examples/telegram_bot.py
async def _run(app: FastAPI, state_dir: Path) -> int: ...  # was _run(state_dir)

# examples/telegram_approval.py
async def _run(app: FastAPI, directory: Path, one_way: bool) -> int: ...
```

Each gets a sync builder alongside it, called from `main` before `asyncio.run`:

```python
def _build(tmp_path: Path) -> tuple[FastAPI, ...]: ...
```

`tests/test_examples.py` grows only tuple entries:

```python
OFFLINE = (
    "comms_loop.py",          # + new
    "core_unit_of_work.py",
    "host_agent.py",          # + new
    "host_report_fixture.py",
    "hostctl_approval_flow.py",
    "hostd_socket_roundtrip.py",
    "telegram_approval.py",   # + new
    "telegram_bot.py",        # + new
)
```

## Sketches

Illustrative, not the patch.

```diff
--- a/examples/host_agent.py
 ROOT = Path(__file__).resolve().parent.parent
 sys.path.insert(0, str(ROOT))
-sys.path.insert(0, str(ROOT / "tests"))
+sys.path.insert(0, str(ROOT / "packages" / "hostd" / "tests"))
```

```diff
--- a/examples/comms_loop.py
-async def run() -> int:
-    with tempfile.TemporaryDirectory() as tmp:
-        ...
-        app = create_app(settings=settings)      # inside the loop -> RuntimeError
-        transport = httpx.ASGITransport(app=app)
+def _build(tmp_path: Path) -> tuple[FastAPI, Path]:
+    """Build the app OFF the event loop: create_app migrates, and a migration
+    opens a transaction, which the engine refuses on a loop thread."""
+    ...
+    return create_app(settings=settings), proj
+
+
+async def run(app: FastAPI, proj: Path) -> int:
+    transport = httpx.ASGITransport(app=app)
 ...
 def main() -> int:
     with tempfile.TemporaryDirectory() as tmp:
-        return asyncio.run(run())
+        app, proj = _build(Path(tmp))
+        return asyncio.run(run(app, proj))
```

Probed end to end on a scratch copy of `comms_loop.py`: exits 0, prints all five
round-trip steps.

## Shape

```
  main()  [sync thread, no running loop]
    |
    +-- Settings(...)
    +-- create_app() ---> state_database() ---> upgrade_to_head()
    |                                              |
    |                                     db.transaction()
    |                                              |
    |                            _refuse_the_event_loop_thread()  -- OK here,
    |                                                                raises if
    |                                                                called below
    +-- asyncio.run( run(app, ...) )   <-- the loop starts only now
                        |
                        +-- httpx / TestClient traffic against the built app


  sys.path for the two fixture importers:

    ROOT/                        ROOT/packages/hostd/tests/
      tests/          -- X -->     test_host_actions.py   (moved by 6d998c8)
      (no such module)               host_files(), host_runner()
```

## Consequences and open questions

- Cost: ~30 lines across four examples plus four tuple entries. No production
  code moves.
- The `OFFLINE` additions make `nix flake check` slower - four more subprocesses,
  each booting an app and (for two) a real unix socket. Measured wall clock for
  the four is a few seconds; the existing timeout is 120s per script and is
  enough.
- The hoist gives up nothing: the app was never built concurrently with anything.
  It does mean each example's `main` now carries a little setup that used to sit
  next to its use. The `_build` docstring is what stops the next person moving it
  back.
- Repointing `sys.path` at `packages/hostd/tests` keeps examples depending on a
  TEST tree's layout, so the next carve can break them the same way - but the new
  `OFFLINE` entries mean it breaks LOUDLY, in CI, which is the property the DoD
  asks for. The alternative - promoting `host_files`/`host_runner` and their
  canned host output into a shipped `scufris_hostd.fakes` module - was rejected
  for this task: it moves several hundred lines of test corpus into the wheel to
  buy a fragility the gate already covers. `examples/hostd_socket_roundtrip.py`
  already reaches into `packages/*/src` by path, so the pattern is house style.
- Open, non-blocking: `host_action.py` and `host_digest.py` also run green and
  look offline, but are not in `OFFLINE`. Adding them is in the spirit of Step 3
  and costs two lines; it is not needed for any DoD proof. Proposed: add them,
  after confirming neither touches the network.
- Not open: `host_inspect.py` and `nixos_change.py` exit 0 here only because this
  box IS a NixOS machine. They stay out of `OFFLINE`. `auth_session.py` binds a
  real port and stays out too. `telegram_bot.py`, despite the name, uses a mock
  backend and no token, so it is genuinely offline - verified by reading, and by
  its failure being the loop guard rather than a network call.
- Verification caveat carried from TASK.md Notes: a stale `.venv` fakes five
  extra failures with `No module named 'scufris_hostctl'`. `uv sync` first. All
  numbers above are post-sync.
