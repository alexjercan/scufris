# Make the config-change restart proofs reopen the database and cover the reap bound

- PRIORITY: 40
- TAGS: test, storage, v0.2.0
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
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

## Definition of Done

- Both restart proofs read through an engine opened after the first app is
  closed (cmd: `python -m pytest tests/test_nixos_config_change.py -k restart`).
- The `MAX_CHANGES` bound is exercised at the store level, settled rows dropping
  ahead of building ones.
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
