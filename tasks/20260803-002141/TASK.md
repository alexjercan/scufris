# Move the configuration-change registry onto the database

- PRIORITY: 70
- TAGS: refactor, v0.2.0, storage, reliability
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a Scufris operator, I want the configuration-change registry on the same
transactional boundary as every other app-owned store, so that a restart during
a NixOS build does not answer "there was never any such change".

## Steps

Ordered. Steps 1-2 are red-first; every later step keeps the whole suite green.
`HostActionStore` (`scufris/host_actions.py`) is the worked example throughout -
row model, one revision, one `db.transaction()` per method.

- [ ] **The failing proofs.** In `tests/test_nixos_config_change.py`, add
      `test_a_configuration_change_survives_a_restart`: build a change through
      `POST /api/host/config/changes`, settle it to `proposed`, rebuild the app
      from the same state directory, and assert
      `GET /api/host/config/changes/{id}` still answers with the state, the
      `toplevel` and the `action_id`. It fails on the base with a 404 - the
      registry went with the process. Add
      `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`: leave a
      change in `building` (the `_BuildExecutor(hang=True)` fixture is already
      there), restart, and assert the restarted app reports that change as
      `failed` with an error naming the restart AND accepts a new change for the
      same repo with 201. On the base this passes vacuously, because nothing
      survives; it is the proof that durability does not introduce a permanent
      409 (DECISION.md 2). In `tests/test_db_state_boundary.py`, delete the
      `not_yet_migrated` exclusion (line 215) and its paragraph in the
      docstring, and add `config_changes` to the asserted store floor.
- [ ] **The schema.** Add `ConfigChangeRow` to `scufris/db/models.py`:
      `id` PK, `seq` unique, `resolved` (JSON text), `attr`, `state`,
      `toplevel`, `action_id`, `run_id`, `log_tail`, `error`, `created_at`,
      `agent`, `requested_by`. `resolved` is JSON text for the same reason
      `HostActionRow.proposal` is - it is a nested model, and nothing queries
      inside it. `seq` is the list ORDER, assigned inside the inserting
      transaction as `max(seq) + 1`, exactly as `HostActionRow` documents.
      Autogenerate ONE revision under `scufris/db/migrations/versions/` with the
      maintainer loop in `scufris/README.md` section 9.
      `test_schema_has_no_pending_autogenerate_diff` is the check.
- [ ] **`ConfigChangeStore` onto the core** (`scufris/hostconfig/changes.py`).
      Constructor takes a `Database`; `self._changes`, the `OrderedDict` import
      and the in-memory `_reap` go. `put` becomes an upsert in one transaction
      (insert with a fresh `seq`, or update every mutable column of an existing
      row and keep its `seq`) - it is the write-back seam the builder uses, so
      it must accept a change it has already stored. `get`, `list` (newest
      first, `ORDER BY seq DESC`) and `building_for` each open one transaction;
      `building_for` selects the `building` rows and matches `resolved.repo` in
      Python rather than adding a column that duplicates a field of `resolved`.
      `_reap` holds `MAX_CHANGES` on the open connection, dropping non-building
      rows first for the same reason `HostActionStore._reap` drops decided ones
      first. Add `abandon_builds()`: one transaction that moves every `building`
      row to `failed` with an error saying the server restarted while it was
      building - see DECISION.md 2.
- [ ] **The builder writes back** (`scufris/hostconfig/changes.py`). Today
      `ConfigChangeBuilder.stream` mutates the `ConfigChange` the store handed
      out and the store sees it because it is the same object; a row does not
      work that way. Add a `save: Save` parameter beside the existing `propose`
      callback (`Save = Callable[[ConfigChange], None]`) and call it after every
      state transition: the `check_attr` refusal, the cancellation (before the
      re-raise), each build failure, the missing store path, the failed
      proposal, and the final `PROPOSED` with its `toplevel` and `action_id`.
      DECISION.md 1 records why this is a callback rather than the store itself.
- [ ] **Wire it up** (`scufris/app.py`). `ConfigChangeStore(db)` at :1742;
      `abandon_builds()` once at startup, beside the `sessions.prune` sweep at
      :1104-1107 and for the same reason. Build the `ConfigChange` with its
      `run_id` already set (`f"config:{cid}"` from the id generated at :1799)
      so the record is stored once rather than put and then mutated at :1806.
      `_stream` passes `config_changes.put` as `save`. Every store call in an
      `async def` route is offloaded with `asyncio.to_thread` -
      `Database.transaction()` refuses a thread with a running loop. Grep
      `rg -n "config_changes\." scufris/` and account for every hit: `get` in
      `_config_change_or_404` (:1755, called from three async routes), `put`
      (:1799), `building_for` (:1786), `list` (:1842). A missed one is a 500 on
      the propose path.
- [ ] **The legacy docstring.** `scufris/db/legacy/__init__.py` says host
      actions have no legacy source because the store was memory-only. Config
      changes are the same case; say so in the same paragraph, so the absence of
      a `config_changes.json` loader reads as decided rather than forgotten.
- [ ] **Docs.** `scufris/README.md` section 9 lists what lives in the state
      database; add the `config_change` table and the restart sweep. Correct any
      sentence in `scufris/README.md` or `scufris/hostconfig/__init__.py` that
      still calls the registry in-memory or bounded-in-process (the module map
      row for `changes` says "the bounded registry").

## Definition of Done

- A configuration change and its proposal survive a restart
  (test: `test_a_configuration_change_survives_a_restart`).
- A build interrupted by a restart is reported failed and does not block the
  next build of that repository
  (test: `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`).
- Every app-owned store shares the declared boundary, with no exclusions left
  (test: `test_post_host_state_uses_declared_persistence_boundary`).
- The boundary test has no named exclusion
  (cmd: `! rg -n 'not_yet_migrated' tests/test_db_state_boundary.py`).
- The registry keeps no in-process dictionary
  (cmd: `! rg -n 'OrderedDict' scufris/hostconfig/changes.py`).
- The schema and the revision agree
  (test: `test_schema_has_no_pending_autogenerate_diff`).
- The absent legacy source is stated rather than inferred
  (cmd: `rg -ni 'configuration change' scufris/db/legacy/__init__.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Found by review round 1 (R1.2) of 20260801-100413, by the boundary test's
  discovery walk rather than by its hand-written store list.
- `ConfigChangeStore` (`scufris/hostconfig/changes.py`) is still an in-memory
  bounded `OrderedDict` with a `_reap`, which is exactly the shape
  `HostActionStore` had before 20260801-100413 migrated it. That task is the
  worked example: a row model, one Alembic revision, one `db.transaction()` per
  method, and `asyncio.to_thread` at every `async def` call site.
- It is app-owned state reached from `app.state.config_changes`, so
  `test_post_host_state_uses_declared_persistence_boundary` in
  `tests/test_db_state_boundary.py` currently EXCLUDES it by name. Removing that
  exclusion is this task's proof - the test then covers every store with no
  exceptions left.
- There is no legacy JSON source: the store was memory-only, so this is the
  same "no legacy file" case host actions were, and
  `scufris/db/legacy/__init__.py`'s docstring should say so for both.
- Out of scope for 20260801-100413, whose Steps name auth, host actions, the
  schedule and the digest history only. Migrating a fifth store was not planned
  there and would have materially exceeded that plan.
- Discovered while planning, and the largest difference from the host-action
  migration: `ConfigChangeBuilder.stream` MUTATES the stored `ConfigChange`
  in place (`changes.py:153,196,205,222,230`) and the store returns the same
  object, so the existing HTTP tests observe those mutations for free. Against a
  row they observe nothing. `tests/test_nixos_config_change.py::_settle` polls
  the GET route until the state leaves `building`, so every build test in that
  file is already the regression net for a missed write-back - a missed one
  hangs for ten seconds and fails.
- Also discovered: durability removes the implicit clear a restart used to give.
  A change left `building` by a crash would survive as `building` forever, and
  `building_for` would then refuse every later build of that repository with a
  409 that `POST .../cancel` cannot clear (it requires a live supervisor run).
  The startup sweep in Step 3 is what keeps that from being a regression, and is
  the reason the second new test exists. DECISION.md 2.
- The propose route's `building_for`-then-`put` is a read-check-write across two
  transactions, so it can still race. Deliberately left as is: the real guard is
  the supervisor's `serialize_key=f"config:{repo}"` (app.py:1834), the 409 is
  documented as "the visible half", and making it one transaction is a change to
  the concurrency contract rather than to where the state lives.
- Assumption: the config-change registry keeps the same `MAX_CHANGES = 100`
  bound. Durability makes the bound matter more, not less, and 100 short-lived
  build records is the same order as the 200 host actions already kept.
