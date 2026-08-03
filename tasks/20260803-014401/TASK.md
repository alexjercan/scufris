# Make the config-change restart proofs reopen the database and cover the reap bound

- PRIORITY: 40
- TAGS: test, storage, v0.2.0
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a Scufris maintainer, I want the restart proofs to actually reopen the
database and the durable `_reap` bound to be exercised, so that "survives a
restart" and "stays bounded" are properties something walks rather than
properties a docstring asserts.

## Notes

- Found by review round 1 of 20260803-002141 (R1.1, R1.2), APPROVEd as MINOR
  because neither falsifies the migration itself. Re-derived independently at
  compound time.
- R1.1: `test_a_configuration_change_survives_a_restart` and
  `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`
  (`tests/test_nixos_config_change.py:566,600`) keep the first `TestClient` open
  across the "restart". `create_app` takes its handle from the process-wide memo
  (`scufris/db/__init__.py:45` `_HANDLES`), so the restarted app is handed the
  SAME `Database` and pool. They prove the registry is no longer a per-app dict
  - both are red on the base - but not that the row is committed and readable
  through a freshly opened engine, which is what the docstrings' "outlive the
  process" claims. Exit the first client, or call `close_state_database`, before
  building the restarted app.
  `test_the_digest_store_survives_a_restart_and_stays_bounded`
  (`tests/test_host_digest.py:165`) is the repo's pattern and says why:
  "Reopened rather than shared".
- R1.2: `ConfigChangeStore._reap` (`scufris/hostconfig/changes.py:174`) is now
  SQL and nothing exercises it; `max_changes` has no caller but the default. A
  silent no-op would grow `config_change` without bound and no check would
  notice. Add a store-level test with `max_changes=3` asserting a settled change
  drops before a building one, and that the oldest goes when all are building.
- R1.3 (NIT, optional): `abandon_builds()` returns a count its only caller
  (`scufris/app.py:1755`) discards and no test asserts. Drop the return per
  YAGNI, or log it at startup so the sweep is observable.
- Scope is tests plus at most the R1.3 one-liner. No behaviour change intended;
  if reopening turns a green test red, that is the finding.
- R1.3 is taken as "drop the return", not "log it at startup". YAGNI: nothing
  reads it, and the sweep already writes a per-row `error` an operator sees.
  Startup observability is its own task if it is wanted.
- The store test lands in `tests/test_nixos_config_change.py`, not a new
  `tests/test_hostconfig_changes.py`. There is no store-level module for this
  domain and one test does not justify creating one; the module docstring's
  "Two layers" becomes three.
- `create_app` publishes its handle as `app.state.db` (`scufris/app.py:264`),
  so identity of that object is the reopen proof and needs no private import.
  Probed on the base: with the first `TestClient` still open the two apps get
  the SAME `Database` (`first.state.db is second.state.db` -> True); with the
  first client exited they differ. That identity assertion is what makes the
  restart proofs red on the base.
- The autouse `_close_process_wide_databases` fixture (`tests/conftest.py:105`)
  only runs at teardown, so it does not help mid-test; the first app's own
  lifespan shutdown is what calls `close_state_database`
  (`scufris/app.py:242`), which closes AND evicts from `_HANDLES`.
- `make_client` cannot express "this one closes early" - its `ExitStack`
  unwinds at teardown - so the first client becomes a literal
  `with TestClient(...)` block. `make_client`'s reason (keep the event loop
  alive between requests) still holds inside that block.
- `abandon_builds()` has exactly one caller, `scufris/app.py:423`, which
  already discards the value, so the call site needs no edit.
- Assumption: reopening keeps both proofs green. The suspect is `action_id`,
  written through the host action store on the same handle. If the reopen turns
  an assertion red, stop and report it rather than weakening the assertion.

## Steps

1. `tests/test_nixos_config_change.py` -
   `test_a_configuration_change_survives_a_restart`: move the pre-restart
   client into `with TestClient(_app(tmp_path, fake_collector, helper,
   config_repo)) as client:`, keeping the POST and `_settle` inside it. Hold
   the app object so the reopen can be asserted; after the block, build the
   restarted app and assert its `state.db is not` the first app's. Existing
   assertions unchanged. Comment why the first process must end first.
2. Same restructuring for
   `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`, whose first
   app carries `build=_BuildExecutor(hang=True)`. Its shutdown cancels the
   in-flight run through `runs.aclose()` and leaves the row `building`, which is
   the state the restarted app's startup sweep must find - confirm the row is
   still `building` inside the block before exiting it.
3. Run `python -m pytest tests/test_nixos_config_change.py -k restart`. If an
   assertion other than the new identity one goes red, stop: that is the
   finding this task exists to surface, and it gets recorded rather than
   patched around.
4. `tests/test_nixos_config_change.py` - add
   `test_the_change_registry_stays_bounded(database: Database)` using the
   conftest `database` fixture and `ConfigChangeStore(database, max_changes=3)`.
   Put four changes with distinct ids; make the oldest `proposed` (settled) and
   the rest `building`, then assert the settled one is the row `_reap` drops
   while every building row survives. Then, with all four `building`, assert
   the oldest `seq` is the one that goes. Import `ConfigChangeStore`,
   `ChangeState`, `ConfigChange`, `Resolved` and `Database` as needed.
5. `scufris/hostconfig/changes.py` - `abandon_builds(self) -> int` becomes
   `-> None`; drop `return result.rowcount` and the "Returns how many" clause
   from the docstring. Leave `scufris/app.py:423` alone.
6. Update the module docstring of `tests/test_nixos_config_change.py` to name
   the third layer (the STORE, against a file-backed database, for the bound).
7. Run `ruff check . && mypy . && python -m pytest`, then `tatr check`.

## Definition of Done

- Both restart proofs build their restarted app only after the first app's
  lifespan has closed and evicted the handle, and assert the restarted app
  holds a different `Database` (cmd: `python -m pytest
  tests/test_nixos_config_change.py -k restart`; red on the base, where the
  two apps share one handle).
- The `MAX_CHANGES` bound is exercised at the store level, settled rows
  dropping ahead of building ones and the oldest going when all are building
  (cmd: `python -m pytest tests/test_nixos_config_change.py -k stays_bounded`;
  red on the base, which collects no such test).
- `abandon_builds` returns nothing (cmd: `grep -n "def abandon_builds(self) ->
  None" scufris/hostconfig/changes.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- Task records lint (cmd: `tatr check`).
