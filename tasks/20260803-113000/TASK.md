# Prove the startup sweep clears a building row orphaned by a crash

- PRIORITY: 35
- TAGS: test, storage, nixos, v0.2.0
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a Scufris maintainer, I want a proof that a `building` row left behind by a
process that died without running its shutdown hooks is swept at the next
startup, so that `abandon_builds` is answered by the state it actually exists
for rather than by a live process that is still building.

## Notes

- Seeded by 20260803-014401 DECISION.md 1, and required by review round 1
  (R1.1) of that task.
- `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`
  (`tests/test_nixos_config_change.py:619`) is green for a reason its docstring
  does not give. Its first `TestClient` is never exited, so the hanging build
  is still hanging and the row is still `building` when the restarted app
  sweeps it. It proves the sweep clears a row some OTHER live process is
  building, not one orphaned by a dead process.
- A clean shutdown cannot produce the target state: the build generator's
  cancellation handler (`scufris/hostconfig/changes.py:329`) catches
  `CancelledError`/`GeneratorExit`, writes `state = CANCELLED` and saves BEFORE
  re-raising, so `Supervisor.aclose()` leaves `cancelled` rows, which the sweep
  neither touches nor needs to.
- Preferred shape, per that record's second alternative: let the first app's
  lifespan close normally, then re-establish a `building` row directly through
  `ConfigChangeStore` against the same state directory - that is what a SIGKILL
  leaves - and build the restarted app. Comment WHY the clean path cannot
  produce the row, so the next reader does not "simplify" it back to an HTTP
  build.
- Keep `test_a_build_interrupted_by_a_restart_does_not_block_the_repo` as is;
  it covers the live-process case and the repo-unblocked follow-through. This
  is an addition, not a replacement.
- Scope is tests only. `abandon_builds` is already correct; no production
  change is expected.
- Correction to the mechanism above, probed in scratch: the lifespan does NOT
  close the config supervisor. `scufris/app.py:236` calls `runs.aclose()`, which
  closes the AGENT supervisor (`scufris/orchestrator/runs.py:496`);
  `config_supervisor_` (`scufris/app.py:417`) is never aclosed. The row still
  ends `cancelled` because the `TestClient` teardown tears down the portal and
  its loop, which cancels the build task and runs the generator's
  `CancelledError`/`GeneratorExit` handler. The CONCLUSION stands - a clean
  shutdown leaves `cancelled` - and the test asserts it rather than assuming it.
- Probed on the base, in a scratch test since removed:
  - after `with TestClient(first)` exits, the row is
    `ChangeState.CANCELLED` with the "stopped before it finished" error;
  - `open_database(tmp_path)` reaches the same file (`_settings` sets
    `state_dir=tmp_path`, `tests/conftest.py:381`) and is safe once the first
    app's lifespan has closed and evicted the process-wide handle;
  - forcing that row back to `BUILDING` with `error=""` and building a second
    app leaves it `failed` with a `restart` reason and `action_id == ""`;
  - with `config_changes.abandon_builds()` (`scufris/app.py:423`) removed, that
    same test fails. The proof is answered by the sweep and by nothing else.
- `error=""` as well as the state: a SIGKILL runs no handler, so the crashed row
  carries no cancellation message. Leaving the cancel error on it would let the
  `"restart" in swept["error"]` assertion pass on a string the sweep never
  wrote.
- The re-established row is the one the real pipeline produced - real `resolved`
  rev, real seq - read back through `ConfigChangeStore.get`; only `state` and
  `error` are moved to where the kill would have left them.
- No repo-unblocked follow-through here (a new 201 after the sweep):
  `test_a_build_interrupted_by_a_restart_does_not_block_the_repo` already owns
  it, and duplicating it would make this test about two things.

## Steps

1. [ ] `tests/test_nixos_config_change.py` - extend the existing
   `from scufris.db import Database` to also import `open_database`.
2. [ ] `tests/test_nixos_config_change.py` - add
   `test_a_building_row_orphaned_by_a_crash_is_swept_at_startup` directly after
   `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`, taking
   `tmp_path`, `fake_collector`, `helper`, `make_client` and `config_repo`.
   Shape:
   - a literal `with TestClient(_app(..., build=_BuildExecutor(hang=True))) as
     client:` block that logs in, POSTs `ref="config/add-ripgrep"`, keeps the
     id, and asserts the change reads `building`; the block exits, so the first
     process ends the way a clean shutdown does;
   - `open_database(tmp_path)` in a `try/finally` that closes it: assert the row
     came back `ChangeState.CANCELLED` (this is what says a clean shutdown
     cannot produce the target state), then set `state = ChangeState.BUILDING`,
     `error = ""` and `put` it - that is what a SIGKILL leaves;
   - `make_client(_app(...))` with the default executor, `_login`, and assert
     the change is now `failed`, that `"restart" in error`, and that
     `action_id == ""`.
   Docstring: what a killed process leaves and why the sweep is the only thing
   that can clear it. Comment WHY the row is re-established rather than built
   over HTTP, naming this task and 20260803-014401 DECISION.md 1, so the next
   reader does not "simplify" it back.
3. [ ] Leave `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`
   byte-identical; it owns the live-process case.
4. [ ] Run `ruff check . && ruff format --check . && mypy . && python -m pytest`,
   then `tatr check`.

## Definition of Done

- A `building` row left by a process that ran no shutdown handler is `failed`
  with a restart reason and no proposal after the next startup (cmd: `python -m
  pytest tests/test_nixos_config_change.py -k orphaned`; red on the base, which
  collects no such test).
- That proof is answered by the sweep and not by the harness: with
  `config_changes.abandon_builds()` removed from `scufris/app.py`, it fails
  (manual: delete that line, run `python -m pytest
  tests/test_nixos_config_change.py -k orphaned`, revert immediately).
- The live-process proof still passes (cmd: `python -m pytest
  tests/test_nixos_config_change.py -k restart`).
- The diff of `tests/test_nixos_config_change.py` changes one import line and
  adds one test function, nothing else (manual: `git diff master --
  tests/test_nixos_config_change.py`).
- All Python checks pass (cmd: `ruff check . && ruff format --check . && mypy .
  && python -m pytest`).
- Task records lint (cmd: `tatr check`).
