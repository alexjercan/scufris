# Add the SQLAlchemy transactional engine core

- STATUS: OPEN
- PRIORITY: 83
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100405

## Story

As a Scufris operator, I want the transaction boundary chosen by the spike to
exist and be proven on its own, so that the three tasks that put real state on
it never debug the boundary and the store at the same time.

## Steps

- [ ] Write the failing proofs first, against a scratch table the core creates
      for the test only: a concurrent burst from threads and loop callbacks, a
      rollback, and a pragma assertion on a second pooled connection.
- [ ] Add `sqlalchemy>=2.0` with `uv add`, `uv lock`, re-enter `nix develop`,
      and confirm the uv2nix closure still builds. SQLAlchemy ships `py.typed`;
      add a `pyproject.toml` mypy override only if a transitive import is
      actually unstubbed. Alembic is task 2's dependency, not this one's.
- [ ] Create `scufris/db/` with `engine.py` holding the whole public surface:
      the engine factory, the pragma hook, and `transaction()`. Keep the
      package under the 600-line source cap from the start - `models.py`,
      `migrations/` and `legacy.py` arrive in the next three tasks.
- [ ] Open one SQLite database at `<state_dir>/scufris.db` and set file mode
      0600 on it and on its `-wal` and `-shm` siblings. The boundary will hold
      auth session identifiers, which `scufris/auth/store.py` protects the same
      way today.
- [ ] Apply DECISION.md section 1's four pragmas through
      `event.listens_for(engine, "connect")` so EVERY pooled connection gets
      `journal_mode=WAL`, `synchronous=FULL`, `busy_timeout=5000` and
      `foreign_keys=ON` - not once at open, which is the failure mode a pool
      introduces over a hand-rolled connection.
- [ ] Make `BEGIN IMMEDIATE` the transaction rather than pysqlite's implicit
      deferred begin: construct the engine with `isolation_level=None` and
      override begin via `event.listens_for(engine, "begin")` calling
      `exec_driver_sql("BEGIN IMMEDIATE")`. Prove it - a deferred begin is
      silently accepted and only fails under contention.
- [ ] Expose exactly one public entry point, the synchronous context manager
      `db.transaction()` over `engine.begin()`. No async engine, no
      `aiosqlite`, no second store API.
- [ ] Give loop-thread callers `asyncio.to_thread` over a synchronous unit of
      work, and state the rule that a transaction never spans an `await`.
- [ ] Make damaged not empty: `sqlalchemy.exc.DatabaseError` on open
      propagates. No tolerant loader anywhere in the package.
- [ ] Add the pytest fixture the next three tasks build on: a file-backed
      database under `tmp_path` with the production pragmas. Not `:memory:` -
      restart-survival proofs must reopen the file, and the measured ~10ms cost
      is affordable.
- [ ] Record the core's public API and its rules in `scufris/README.md`, with
      the `scufris.db` module map, so the follow-ups do not re-derive them.

## Definition of Done

- A failed multi-statement unit of work commits nothing
  (test: `test_state_transaction_rolls_back_as_a_unit`).
- A thread-pool caller and a loop callback offloading through
  `asyncio.to_thread` mutate concurrently without loss
  (test: `test_sync_and_async_callers_share_the_transaction_boundary`).
- Every pooled connection carries the four production pragmas, not just the
  first (test: `test_every_pooled_connection_applies_production_pragmas`).
- The transaction takes the write lock up front rather than deferring
  (test: `test_transaction_uses_begin_immediate`).
- A damaged database raises instead of presenting itself as empty
  (test: `test_damaged_state_refuses_to_load`).
- The database and its `-wal`/`-shm` siblings are owner-only
  (test: `test_state_database_files_are_owner_only`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- The Nix closure still builds with the new dependency
  (cmd: `nix build .#scufris`).

## Notes

- Epic: 20260729-102145. Lane B, first of four.
- OVERRIDES 20260801-100405 DECISION.md sections 1 and 4 at the user's
  direction: SQLAlchemy 2.0 + Alembic replace stdlib `sqlite3` + the
  `PRAGMA user_version` ladder. That DECISION.md listed this exact alternative
  and rejected it on cost (two dependencies in `uv.lock` and the uv2nix
  closure) while recording it as "reversible later at low cost". On approval,
  write `tasks/20260729-102147/DECISION.md` recording the override and its
  reason, and mark sections 1 and 4 of the spike's decision SUPERSEDED - do not
  silently diverge from an ACCEPTED record.
- What the override does NOT change: one database, one boundary, the four
  pragmas, `BEGIN IMMEDIATE`, transaction as the read-modify-write boundary, no
  in-memory mirror, damaged-is-not-empty, synchronous API with
  `asyncio.to_thread` offload, file-backed test fixtures, and the whole import
  policy table.
- What it costs: `sqlalchemy` pulls `greenlet` into `uv.lock` and the uv2nix
  closure; the pragma and `BEGIN IMMEDIATE` recipes become event hooks rather
  than two lines at connect.
- No store moves here and no product schema is created. `projects.json` and
  every other JSON store keep working untouched, which is what lets this land
  green on its own.
- Preserve the local single-host deployment model. SQLAlchemy's portability is
  a side effect, not the reason.
