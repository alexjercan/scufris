# Notes: Make the config-change restart proofs reopen the database and cover the reap bound

## What changes

Nothing an operator sees. This is test-only, plus one optional signature
cleanup.

Before:

- `test_a_configuration_change_survives_a_restart` and
  `test_a_build_interrupted_by_a_restart_does_not_block_the_repo` build a
  second app while the first `TestClient` is still open. `create_app` takes its
  handle from the process-wide memo (`scufris/db/__init__.py:45` `_HANDLES`),
  and the first app's lifespan has not run its `close_state_database`
  (`scufris/app.py:242`) yet, so the "restarted" app is handed the SAME
  `Database` object and the SAME connection pool. The tests are red on the base
  and do prove the registry is no longer a per-app dict, but they do not prove
  what their docstrings claim - that the rows are committed and readable
  through a freshly opened engine.
- `ConfigChangeStore._reap` (`scufris/hostconfig/changes.py:174`) is SQL that
  nothing exercises. `max_changes` has no caller but the `MAX_CHANGES` default.
  A silent no-op grows `config_change` without bound and no check notices.
- `abandon_builds()` returns a count its only caller (`scufris/app.py:423`)
  discards, and no test asserts it.

After:

- Both restart proofs let the first app's lifespan run to completion - closing
  AND evicting the handle - before the restarted app is built, so the second
  app opens the file again. Same assertions, now backed by a real reopen.
- A store-level test drives `_reap` with `max_changes=3` and pins its two
  documented rules: settled changes go before building ones, and when
  everything is building the oldest goes anyway.
- `abandon_builds()` loses its unused return (R1.3, NIT).

If reopening turns a currently green assertion red, that is the finding and the
task stops for it rather than papering over it.

## Surfaces

| File | Why |
|------|-----|
| `tests/test_nixos_config_change.py` | the two restart proofs; add the store-level reap test here, as a third layer beside the app and repository layers the module docstring already names |
| `scufris/hostconfig/changes.py` | R1.3 only: `abandon_builds() -> int` becomes `-> None` |
| `scufris/app.py` | R1.3 follow-through if the call site needs no change (it already discards the value, so likely untouched) |

No production behaviour, no schema, no HTTP surface.

## Data and interfaces

Test-side only:

- `test_a_configuration_change_survives_a_restart(...)` - unchanged signature.
  Body restructured so the pre-restart client lives in its own
  `with TestClient(...)` block.
- `test_a_build_interrupted_by_a_restart_does_not_block_the_repo(...)` - same
  restructuring. Note the interrupted build is `hang=True`; the first app's
  shutdown cancels the in-flight run through `runs.aclose()` and leaves the row
  `building`, which is exactly the state the restarted app's startup sweep is
  supposed to find.
- `test_the_change_registry_stays_bounded(database: Database) -> None` - new,
  uses the existing conftest `database` fixture (file-backed, at head) and
  `ConfigChangeStore(database, max_changes=3)`.

Production-side, R1.3 only:

```python
def abandon_builds(self) -> None:  # was -> int
```

## Sketches

Illustrative, not the patch.

Restart proof (`tests/test_nixos_config_change.py`):

```diff
-    client = make_client(_app(tmp_path, fake_collector, helper, config_repo))
-    csrf = _login(client)
-    ...
-    before = _settle(client, csrf, resp.json()["id"], want="proposed")
+    # The first process ends before the second starts: leaving the client open
+    # would leave `create_app`'s handle in the process-wide memo, and the
+    # "restarted" app would be handed the same Database and pool.
+    with TestClient(_app(tmp_path, fake_collector, helper, config_repo)) as client:
+        csrf = _login(client)
+        ...
+        before = _settle(client, csrf, resp.json()["id"], want="proposed")

     restarted = make_client(_app(tmp_path, fake_collector, helper, config_repo))
```

Store-level bound:

```diff
+def test_the_change_registry_stays_bounded(database: Database) -> None:
+    store = ConfigChangeStore(database, max_changes=3)
+    ...
+    # A settled change goes before a building one - the building one has a live
+    # run behind it.
+    assert [c.id for c in store.list()] == [...]
```

## Shape

```
  test_a_configuration_change_survives_a_restart

    with TestClient(_app(...)) as client:        <- "process 1"
         |  POST /api/host/config/changes
         |  _settle -> proposed
    exit +--> lifespan shutdown
              runs.aclose()
              close_state_database(state_dir)    <- closes AND evicts _HANDLES

    make_client(_app(...))                       <- "process 2"
         |  state_database(state_dir): memo MISS
         |  open_state_database -> new engine, new pool, reads the file
         v  GET /api/host/config/changes/<id> -> proposed, toplevel, action_id


  test_the_change_registry_stays_bounded          (no app, no HTTP)

    ConfigChangeStore(database, max_changes=3)
         put() -> _reap(conn) inside the SAME transaction
                   order by (state == building) asc, seq asc
                   delete the oldest `over` rows
```

## Consequences and open questions

Costs:

- The restart tests get one more level of indentation and stop using
  `make_client` for their first client. `make_client`'s docstring reason -
  hold the client open so the event loop survives between requests - still
  holds inside the `with` block, so nothing is lost; the fixture just cannot
  express "this one closes early", because its `ExitStack` unwinds at teardown.
- Each restart test now pays a real second `open_state_database` (open,
  migrate, legacy import). Single-digit milliseconds, per the `database`
  fixture's own note.

Forecloses nothing. No production path changes except the R1.3 return type.

Open questions, all recorded as assumptions rather than blocking:

- The store test lands in `tests/test_nixos_config_change.py` rather than a new
  `tests/test_hostconfig_changes.py`. There is no store-level module for this
  domain today and one test does not justify creating one; the module docstring
  gets a line for the new layer.
- R1.3 is taken as "drop the return" rather than "log it at startup". YAGNI:
  nothing reads it, and the sweep already writes a per-row `error` an operator
  can see. If the sweep should be observable at startup that is its own task.
- Whether reopening keeps both proofs green is genuinely unknown until it runs.
  The suspect is `action_id`: the proposal's action is written through the host
  action store on the same handle, so if anything there is memoized per-app
  rather than per-row the reopen will surface it. That is the point of the
  change.
