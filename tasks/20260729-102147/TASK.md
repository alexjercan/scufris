# Add the SQLAlchemy transactional engine core

- PRIORITY: 83
- TAGS: bug, v0.2.0, reliability, storage, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100405

## Story

As a Scufris operator, I want the transaction boundary chosen by the spike to
exist and be proven on its own, so that the three tasks that put real state on
it never debug the boundary and the store at the same time.

## Steps

- [x] Write the failing proofs first, against a scratch table the core creates
      for the test only: a concurrent burst from threads and loop callbacks, a
      rollback, and a pragma assertion on a second pooled connection.
- [x] Add `sqlalchemy>=2.0` with `uv add`, `uv lock`, re-enter `nix develop`,
      and confirm the uv2nix closure still builds. SQLAlchemy ships `py.typed`;
      add a `pyproject.toml` mypy override only if a transitive import is
      actually unstubbed. Alembic is task 2's dependency, not this one's.
- [x] Create `scufris/db/` with `engine.py` holding the whole public surface:
      the engine factory, the pragma hook, and `transaction()`. Keep the
      package under the 600-line source cap from the start - `models.py`,
      `migrations/` and `legacy.py` arrive in the next three tasks.
- [x] Open one SQLite database at `<state_dir>/scufris.db` and set file mode
      0600 on it and on its `-wal` and `-shm` siblings. The boundary will hold
      auth session identifiers, which `scufris/auth/store.py` protects the same
      way today.
- [x] Apply DECISION.md section 1's four pragmas through
      `event.listens_for(engine, "connect")` so EVERY pooled connection gets
      `journal_mode=WAL`, `synchronous=FULL`, `busy_timeout=5000` and
      `foreign_keys=ON` - not once at open, which is the failure mode a pool
      introduces over a hand-rolled connection.
- [x] Make `BEGIN IMMEDIATE` the transaction rather than pysqlite's implicit
      deferred begin: construct the engine with `isolation_level=None` and
      override begin via `event.listens_for(engine, "begin")` calling
      `exec_driver_sql("BEGIN IMMEDIATE")`. Prove it - a deferred begin is
      silently accepted and only fails under contention.
- [x] Expose exactly one public entry point, the synchronous context manager
      `db.transaction()` over `engine.begin()`. No async engine, no
      `aiosqlite`, no second store API.
- [x] Give loop-thread callers `asyncio.to_thread` over a synchronous unit of
      work, and state the rule that a transaction never spans an `await`.
- [x] Make damaged not empty: `sqlalchemy.exc.DatabaseError` on open
      propagates. No tolerant loader anywhere in the package.
- [x] Add the pytest fixture the next three tasks build on: a file-backed
      database under `tmp_path` with the production pragmas. Not `:memory:` -
      restart-survival proofs must reopen the file, and the measured ~10ms cost
      is affordable.
- [x] Record the core's public API and its rules in `scufris/README.md`, with
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

## Close-out

### What and why

`scufris/db/` now exists, holding the whole boundary and nothing else: 178 lines
of `engine.py` plus a re-exporting `__init__.py`. The public surface is
`open_database(state_dir) -> Database`, `Database.transaction()`,
`Database.engine`, `Database.path` and `database_path()`. No store moved, no
product schema was created, and every JSON store is untouched - which is why the
branch lands green on its own.

`sqlalchemy>=2.0` (2.0.51, pulling `greenlet` 3.5.4) is in `pyproject.toml` and
`uv.lock`. No mypy override was needed: SQLAlchemy ships `py.typed` and
`mypy .` is green on 172 files. Alembic is deliberately absent; it is task
20260801-120404's dependency.

### Alternatives considered

- **`create_engine(isolation_level=None)`, as the Step spelled it.** Not valid
  at the SQLAlchemy level - that parameter takes an isolation-level STRING
  (`"SERIALIZABLE"`, `"AUTOCOMMIT"`), and the thing that has to be `None` is
  pysqlite's own `isolation_level` attribute. Implemented as
  `connect_args={"isolation_level": None}`, which is the same intent through the
  parameter that reaches the driver, and is SQLAlchemy's own documented recipe.
  The Step's outcome is unchanged and proven.
- **A `Database.read()` context manager alongside `transaction()`.** Because
  every begin is immediate, a read-only unit of work also takes the write lock.
  Rejected: DECISION.md's "exactly one public entry point" is worth more than
  the contention a short read costs, and a second entry point is the second
  store API that decision removed. Recorded in the module docstring and in
  `scufris/README.md` section 9 so the follow-ups do not rediscover it as a bug.
- **An `await db.run(fn)` offload helper.** Rejected as a wrapper around one
  stdlib call with no caller yet. The rule is stated instead, in the docstring,
  the README and the test.

### Difficulties and diagnosis

`test_every_pooled_connection_applies_production_pragmas` failed on first run
with "database is locked" rather than a pragma mismatch. Cause was the test, not
the code: reading a pragma through `Connection.execute` opens a transaction, and
every begin on this engine is a `BEGIN IMMEDIATE`, so holding two SQLAlchemy
connections open made them serialise on the write lock. Reading through
`engine.raw_connection()` and the DBAPI cursor asks the connection about itself
without beginning anything. This is the first concrete instance of the
read-takes-the-write-lock consequence above, which is why it is now documented.

### Evidence

All six named proofs pass, plus two extra: reopening the file sees committed rows
(restart survival), and the boundary is usable from a worker thread.

Both pool-specific proofs were FALSIFIED rather than merely observed green:

| Sabotage | What broke |
|-|-|
| `BEGIN IMMEDIATE` -> `BEGIN DEFERRED` | `test_transaction_uses_begin_immediate` DID NOT RAISE, and `test_sync_and_async_callers_share_the_transaction_boundary` died with the real SQLITE_BUSY-on-upgrade that `busy_timeout` does not retry |
| pragmas applied once at open instead of on `connect` | `test_every_pooled_connection_applies_production_pragmas` failed on the second connection (`foreign_keys` 0 != 1) |

- `ruff check . && mypy .`: clean.
- `python -m pytest`: 902 passed, 2 failed - `test_project_tasks_endpoint` and
  `test_read_project_tasks_parses_real_tatr`, both failing identically on master
  before this branch (verified by running them in the main checkout). Not caused
  here and out of scope.
- `nix build .#scufris`: builds, so the uv2nix closure absorbs sqlalchemy and
  greenlet.
- `ruff format` was run on the files this branch touches only; the tree has 12
  other pre-existing unformatted files and reformatting them would bury the diff.

### Reflection

The falsification pass earned its cost twice over. Both hooks were green on the
first run, and a green test that would also be green with the hook removed would
have handed the next three tasks a boundary that only looked proven - which is
the exact failure this task exists to prevent.

The one thing worth carrying forward is that a per-connection assertion cannot
be made through the ORM's own connection when every begin is immediate. Any
follow-up asserting connection state needs `raw_connection()`.

Note also that `journal_mode` is a persistent property of the database FILE, not
of the connection, so it is the one pragma of the four that does not discriminate
in that test. The other three do.

### Round 1 addendum

Review round 1 opened thirteen findings, one MAJOR, and all thirteen are fixed on
the branch. The MAJOR is the one worth carrying forward: `transaction()` was not
reentrant and nothing said so, so the natural mistake for the next three tasks -
one store's unit of work calling another store's - would have waited the full
5s busy timeout on the write lock the OUTER transaction was holding and then
failed with a message that reads as external contention. It now raises
`RuntimeError` immediately, guarded by a `ContextVar` so a worker thread starts
with its own copy.

The guard fails FAST rather than making nesting silently work. Reusing the outer
connection on re-entry would give an inner `with` block that appears to commit
and does not, which is a worse failure than the deadlock it replaces.

Three of the thirteen (R1.1, R1.3, R1.4) were gaps between what the docs promised
and what the code did, on the one task whose entire purpose is that the follow-ups
can trust the boundary without re-deriving it: the non-reentrancy was undocumented,
"raises at open" was true only for an unreadable header, and the sidecar chmod
loop never executed on the fresh-open path so its test was really proving SQLite's
mode inheritance. The lesson is that recording an API is not the same as testing
the recorded API against the code.

Four new guards, four new falsifications, all caught:

| Sabotage | What broke |
|-|-|
| nesting guard removed | `test_nested_transactions_are_refused_immediately` - the real 5s `OperationalError` |
| `ContextVar` -> module-level flag | the per-context test plus BOTH existing concurrency tests |
| sidecar loop reduced to `(path,)` | `test_sidecars_left_behind_by_a_crash_are_narrowed_on_open` |
| `O_NOFOLLOW` dropped | `test_a_symlinked_database_path_is_refused` |

Re-verified after the fixes: `ruff check .` and `mypy .` clean, `python -m pytest`
907 passed with the same 2 pre-existing failures (filed as 20260801-123345),
`nix build .#scufris` green, and the file-size ratchet green at 222 lines for
`engine.py` and 371 for its tests.
