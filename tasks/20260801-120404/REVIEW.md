# Review: Land the Alembic migration runner and the projects schema

- TASK: 20260801-120404
- BRANCH: fix/alembic-migration-runner

## Round 1

- REVIEWER: out-of-context (three lanes: behavior/proofs, correctness/security/concurrency, design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/db/migrate.py:140 - two processes calling
  `migrate_state_dir` on a FRESH state dir crash one of them BEFORE the runner's
  write lock is reached, so the module docstring's "two processes starting
  together cannot both decide to create the same table... the second waits on
  the lock" is not what happens on a first run. `PRAGMA journal_mode=WAL` in
  `open_database` does not honour `busy_timeout`: measured, it raises "database
  is locked" after 0.000s against a 5s timeout, and `scufris/db/engine.py:70-73`
  claims the opposite. Measured with 4 concurrent OS processes: a fresh state dir
  fails on the first trial at `engine.py:212`; a state dir already in WAL mode
  passes 6 trials x 4 processes. Retry the `journal_mode` pragma on
  `OperationalError` with a bounded wait (SQLite never invokes the busy handler
  for a journal-mode change), and correct the `busy_timeout goes FIRST` comment
  to say what it does and does not cover.
  - Response: Fixed. `engine._set_journal_mode` retries `PRAGMA journal_mode=WAL`
    on a contention `OperationalError` for `JOURNAL_MODE_TIMEOUT` (5.0s, matching
    busy_timeout), polling at 0.05s; anything not "locked"/"busy" still raises at
    once. The `connect` hook branches on the pragma. Reproduced the race first,
    then pinned it with
    `tests/test_db_engine.py::test_open_waits_out_a_concurrent_first_wal_conversion`
    (a raw `sqlite3` holder plus a releaser thread); removing the branch fails
    it. The `busy_timeout goes FIRST` comment now states the exception and the
    measured 0.000s.
- [x] R1.2 (MAJOR) scufris/db/migrate.py:121 - the already-at-head path takes the
  exclusive `BEGIN IMMEDIATE` before it can discover there is nothing to do, and
  `busy_timeout=5000` is the only backstop: with the lock held 7s the second
  `BEGIN IMMEDIATE` raises a raw `OperationalError: database is locked` after
  exactly 5.0s. Every startup after the cutover therefore hard-fails if any
  writer holds the lock for over 5s. Add a lock-free pre-check above line 121 -
  read `_current_revision` on a plain `engine.connect()` and return when it
  equals head - keeping the locked check-then-act only for an actual migration,
  and re-raise a lost race with a message naming the database file and "another
  Scufris process is migrating it".
  - Response: Fixed, with one deviation from the suggested mechanism. A plain
    `engine.connect()` does NOT give a lock-free read on this engine: the `begin`
    event makes every begin a `BEGIN IMMEDIATE`, so the suggested pre-check still
    took the write lock - the new test failed "database is locked" against it.
    `current_revision(db)` therefore reads on a raw DBAPI connection
    (`migrate.py:93-117`), which in WAL never blocks and is never blocked, and
    treats "no such table" as never-migrated. `upgrade_to_head` calls it before
    the transaction and returns early; the locked re-read stays for the
    check-then-act. `OperationalError` containing "locked" is re-raised as a
    `RuntimeError` naming `db.path` and the holding process. Pinned by
    `test_a_database_at_head_does_not_take_the_write_lock`, which holds the write
    lock from a raw `sqlite3` connection (a second `Database` in-process trips
    the nesting guard instead, which would have passed for the wrong reason):
    0.09s as shipped, genuine "database is locked" with the pre-check removed.
- [x] R1.3 (MAJOR) scufris/db/migrate.py:107 - `VACUUM INTO` creates the backup
  under the process umask and `os.chmod` at line 114 only narrows it afterwards.
  Measured under `umask 022`: the file exists at 0644 for the whole duration of
  the copy - a complete copy of a database the engine docstring says "will hold
  live session ids", world-readable for as long as the vacuum runs. Wrap the
  statement in `old = os.umask(0o077)` / `os.umask(old)` so SQLite creates it
  0600, and keep the `chmod` as a belt-and-braces narrow.
  - Response: Fixed exactly as described: `os.umask(0o077)` wraps the
    `VACUUM INTO` and is restored in the `finally`; the `chmod` stays. Pinned by
    `test_the_backup_is_never_world_readable_even_briefly`, which monkeypatches
    `os.chmod` to a no-op and asserts the created mode is 0600. (A first attempt
    using `set_progress_handler` to observe the file mid-vacuum never fired on a
    database this small, so it proved nothing and was replaced.) With the umask
    wrap removed the test reports 0o644.
- [x] R1.4 (MAJOR) scufris/mcp_server.py:583 - Step 8's "check it also runs for
  the MCP subprocess entry point" is not pinned by any test.
  `test_the_mcp_subprocess_upgrades_the_same_database`
  (tests/test_db_migrations.py:266) calls the `migrate_state` seam directly, so
  it re-proves `migrate_state_dir` and deleting line 583 from `main()` fails
  nothing. `migrate_state` is also a one-line wrapper with one production caller
  and a nine-line docstring. Delete the wrapper (scufris/mcp_server.py:553-565),
  call `migrate_state_dir(Settings().state_dir)` inline in `main`, move its
  rationale into the README paragraph that already carries it, and assert the
  revision after `main()` in `test_main_configures_logging_and_runs`
  (tests/test_mcp_server.py:43), which already drives `main()` under an isolated
  state dir - so the call SITE is what is measured.
  - Response: Fixed as described. The `migrate_state` wrapper is deleted; `main()`
    calls `migrate_state_dir(Settings().state_dir)` inline with a three-line note,
    and the rationale lives in `scufris/README.md` section 9.
    `test_the_mcp_subprocess_upgrades_the_same_database` is gone;
    `tests/test_mcp_server.py::test_main_configures_logging_and_runs` now asserts
    `current_revision(db) == head_revision()` after `main()`. Deleting the call
    from `main()` fails that test.
- [x] R1.5 (MINOR) scufris/db/migrate.py:126 - a database written by a NEWER
  build is treated as merely "behind head": the backup is written first and only
  then does Alembic raise `CommandError: Can't locate revision identified by
  'deadbeefcafe'` (measured, leaving `scufris.db.pre-deadbeefcafe.bak` behind).
  The same unvalidated value reaches `backup_path` and the filename. Before
  backing up, resolve `current` through `ScriptDirectory.get_revision`; on a
  miss, refuse with a message saying the database was written by a newer Scufris
  and naming the unknown `alembic_version` row.
  - Response: Fixed. `_known_revision` resolves `current` through
    `ScriptDirectory.get_revision` and `upgrade_to_head` refuses before
    `backup_database` runs, so no stray `.bak` is written; the message names the
    unknown revision and tells the operator to install that version or restore a
    backup taken before it. Pinned by
    `test_a_database_from_a_newer_scufris_is_refused_without_a_backup`, which
    asserts both the raise and the absence of the backup file.
- [x] R1.6 (MINOR) tests/test_db_migrations.py:213 -
  `test_migration_scripts_ship_inside_the_package` and the matching DoD command
  do not discriminate where they run: inside the repo,
  `importlib.resources.files('scufris.db.migrations')` resolves to the WORKTREE
  source, so both pass even if the wheel excluded the files. (The claim itself
  holds - it was verified against the built store path - but the test that is
  supposed to guard it cannot fail.) Assert against the built output instead,
  e.g. resolve the path and assert it is not under the repo root, or make the
  DoD command `nix build .#scufris && test -f result/.../migrations/script.py.mako`.
  - Response: Fixed. DoD proof 4 is now
    `nix build .#scufris && test -f "$(dirname "$(readlink -f result/bin/scufris)")/../lib/python3.14/site-packages/scufris/db/migrations/script.py.mako"`,
    which reads the built store path and cannot be satisfied by the worktree.
    Re-run green this round. `test_migration_scripts_ship_inside_the_package`
    stays as the in-repo sibling check on the package layout, which is what it
    can honestly measure.
- [x] R1.7 (MINOR) tests/test_db_migrations.py:157 -
  `test_a_fresh_database_is_not_backed_up` and
  `test_a_database_at_head_is_neither_migrated_nor_backed_up` (line 164) assert
  only the absence of a `*.bak`, with no delivery guard that the provoking
  stimulus fired; both stay green if `migrate_state_dir` becomes a no-op, and the
  second's docstring claims "no revision" while asserting nothing about one. Add
  `assert current_revision(db) == head_revision()` to both, and for the second
  assert the revision is unchanged across the two calls.
  - Response: Fixed as described. The first now asserts
    `current_revision(db) == head_revision()` as its delivery guard; the second
    reads the revision after the first `migrate_state_dir`, asserts it is head,
    then asserts it is unchanged after the second call. Both docstrings were
    rewritten to say what is actually asserted.
- [x] R1.8 (MINOR) tests/conftest.py:154 - `upgrade_to_head(db)` runs outside the
  `try`, so a migration failure leaks the engine and its file handles for every
  test using the `database` fixture. Move it inside the `try:` block above
  `yield db`.
  - Response: Fixed. `upgrade_to_head(db)` is now the first statement inside the
    `try:`, so the `finally: db.close()` covers a migration failure.
- [x] R1.9 (MINOR) scufris/db/migrations/env.py:41 - `render_as_batch=True` is
  set on a connection running with `foreign_keys=ON` inside an already-open
  transaction, where `PRAGMA foreign_keys` is a no-op. A future batch `ALTER` -
  the copy-and-move this option exists for - therefore cannot turn foreign keys
  off, and once a table with an FK exists it will fail or silently rewrite child
  references. Record the constraint at line 41 as an implementation note ("batch
  ALTERs on this path require FK-free tables; re-decide before the first FK") so
  the next revision does not discover it live.
  - Response: Recorded. The note moved with the options themselves: it now sits on
    `MIGRATION_CONTEXT_OPTS` in `scufris/db/migrate.py` (the single place R1.11
    hoisted them to), saying that `PRAGMA foreign_keys` is a no-op inside the
    open transaction, that batch ALTERs on this path require FK-free tables, and
    that this must be re-decided before the first foreign key lands.
- [x] R1.10 (MINOR) scufris/db/migrate.py:114 - `os.chmod` follows symlinks,
  unlike `engine._secure`, which explicitly refuses a symlinked candidate; the
  `unlink` at line 103 shrinks but does not close the window before
  `VACUUM INTO` and the chmod. Mirror `_secure`: raise if `target.is_symlink()`
  rather than writing and chmod-ing through it.
  - Response: Fixed. `backup_database` raises
    `RuntimeError(f"{target} is a symlink; refusing to write the backup")` before
    the unlink, mirroring `_secure`.
- [x] R1.11 (MINOR) tests/test_db_migrations.py:86 - the drift proof re-types
  `compare_type` and `render_as_batch` by hand instead of reading the options
  `env.py:41` configures, so changing env.py alone silently stops the proof
  matching production. Hoist them to one constant in `scufris/db/migrate.py`
  and have both env.py and the test import it.
  - Response: Fixed. `MIGRATION_CONTEXT_OPTS` lives in `scufris/db/migrate.py`;
    `env.py` and `test_schema_has_no_pending_autogenerate_diff` both import it.
    One deviation: `env.py` passes the two values by key rather than splatting
    `**MIGRATION_CONTEXT_OPTS` - `context.configure` is typed as named keyword
    arguments and a `**dict[str, bool]` fails mypy (11 errors). The VALUES still
    come from the one constant, which is what the drift proof compares under; the
    reason is a comment at the call site.
- [x] R1.12 (MINOR) scufris/db/migrate.py:122 - `upgrade_to_head` opens its own
  write path (`db.engine.connect()` + `conn.begin()`), bypassing
  `Database.transaction()`, which `scufris/db/engine.py:23-41` documents as the
  one write boundary and the only thing carrying the re-entrancy guard. Behavior
  matches (the begin event still yields `BEGIN IMMEDIATE`), the idiom does not.
  Replace the two nested `with` lines with `with db.transaction() as conn:`.
  - Response: Fixed. `upgrade_to_head` now writes through
    `with db.transaction() as conn:`, so it carries the re-entrancy guard like
    every other writer. (Noted for the reviewer: this is also why R1.2's lock-free
    read had to leave SQLAlchemy entirely - the boundary has no read-only mode by
    design.)
- [x] R1.13 (MINOR) alembic.ini:24 - the `[loggers]`, `[handlers]` and
  `[formatters]` sections are dead: `env.py` deliberately never calls
  `logging.config.fileConfig`, so nothing reads them. Delete lines 24-56, or
  restore a `fileConfig` call on the development path.
  - Response: Fixed. The three sections are deleted, with a comment saying the
    file carries no logging config because `env.py` never calls `fileConfig`.
- [x] R1.14 (NIT) scufris/db/migrate.py:52 - `alembic_config` has no caller
  outside its own module; rename it `_alembic_config` so the module's public
  surface is the names the README and the tests actually use.
  - Response: Fixed. Renamed to `_alembic_config`, with no remaining reference in
    code or docs. (Correction after round 2: an earlier wording of this Response
    described the README's `scufris.db` public-surface table as listing
    `head_revision`, `current_revision` and the backup helpers. It does not, and
    should not - those are `scufris.db.migrate` names reached by module import;
    the table's migration rows are `migrate_state_dir` and `upgrade_to_head`.)
- [x] R1.15 (NIT) scufris/app.py:949 - "and today no store depends on it yet -
  the stores are still on JSON until the cutover tasks" is roadmap prose that
  goes stale at the next task; the invariant is the sentence above it. Trim to
  the invariant, and likewise the third paragraph of `scufris/db/models.py`,
  which lists the nine stores still to come.
  - Response: Fixed. The `app.py` comment is trimmed to the invariant (schema up
    before the first store; a no-op at head) and the roadmap paragraph is gone
    from `models.py`. One roadmap sentence survives deliberately, in
    `migrate_state_dir`'s docstring: it explains why that function opens and
    closes its OWN handle, which stops being true at the cutover and should be
    re-read then.
- [x] R1.16 (NIT) AGENTS.md:17 - a new root-level config file, `alembic.ini`, is
  absent from the "Sources of truth" table. Add
  `| alembic.ini | Maintainer-only autogenerate config; workflow in scufris/README.md section 9 |`.
  - Response: Fixed. Row added to the AGENTS.md "Sources of truth" table as
    written.

Verified independently by the primary, not taken from a lane:

- The fresh-state-dir race in R1.1 was found by the primary's own four-process
  probe and is not covered by any lane; the correctness lane recorded real
  cross-process contention as NOT verified.
- Reproduced first-hand: the 0644 backup window (R1.3), the newer-revision
  backup-then-`CommandError` (R1.5), and that `PRAGMA journal_mode=WAL` ignores
  `busy_timeout` entirely (0.000s against 5s).
- Checks rerun by the primary: `ruff check .` clean, `mypy .` clean on 178
  files, `python -m pytest` 922 passed / 2 failed, `nix flake check` all checks
  passed, `nix build .#scufris` builds. The two failures
  (`test_project_tasks_endpoint`, `test_read_project_tasks_parses_real_tatr`)
  shell out to a real `tatr` and fail identically on `master`; confirmed by two
  lanes independently.
- All seven `tatr proofs` pass on their stated criteria. The three
  discrimination claims in the close-out were re-falsified by the behavior lane
  (stub `app.migrate_state_dir`, add a table to `Base.metadata`, force `env.py`
  down the URL path) and all three fail as claimed.
- Close-out honesty: alembic 1.18.5 / Mako 1.3.12 confirmed in `uv.lock`; head
  revision `8f8087f3cc9c` confirmed; the disclosed limitation that the backup has
  no end-to-end exercise is accurate and not hidden.
- No DECISION.md is owed: `tasks/20260729-102147/DECISION.md` already records
  Alembic over the `user_version` ladder and the in-package environment, and
  names this task as an implementer.
- Doc sweep run over README.md, scufris/README.md, AGENTS.md, CHANGELOG.md,
  docs/ and .env.example for every changed symbol, path and command; no stale
  mention found beyond R1.16.

- Process signal: Step 5's literal text says set "the URL from
  `Settings.state_dir`". The shipped runner deliberately sets no
  `sqlalchemy.url`, and the Step was ticked anyway. The close-out discloses and
  argues the deviation soundly - the URL is exactly what would let `env.py` open
  its own engine, contradicting the same Step's other clause - but the plan text
  should have been amended rather than the tick carried on intent.
- Process signal: the branch is a single squashed commit, so Step 1's
  "failing proofs first" ordering is not observable from history.
- Process signal: two of the three most load-bearing proofs in this task passed
  under their own sabotage on first write and had to be rewritten to
  discriminate. That is now twice on this lane (see
  `tasks/20260729-102147/TASK.md`), which suggests the sabotage belongs inside
  the test-writing step rather than in verification.

Process signals, answered by the primary:

- Step 5's URL clause is amended in TASK.md rather than left carried on intent,
  with the deviation noted in place.
- The squash is how this branch lands; the ordering claim is not observable from
  history and is not re-asserted.
- Sabotage moved into the test-writing step for this round: each of the five new
  or rewritten proofs was falsified as it was written, and the table is in the
  close-out. Two of them still failed their first falsification and were
  rewritten before the round ended, which is the point of doing it there.

Round 2 verification (primary): `ruff check .` clean, `mypy .` clean on 178
files, `python -m pytest` 2 failed (the same two pre-existing `tatr`-shelling
tests, unchanged on master), `nix flake check` all checks passed,
`nix build .#scufris` builds, and the amended DoD proof 4 finds
`script.py.mako` under the built store path.

Pending user checks: none. This task carries no `manual:` proof.

## Round 2

- REVIEWER: out-of-context (fresh reviewer, no round-1 context, verdicts
  re-derived by sabotage rather than read off the Responses)
- VERDICT: APPROVE

All sixteen round-1 findings verified fixed. Each of R1.1, R1.2, R1.3, R1.4,
R1.5 and R1.7 was confirmed by sabotaging the fix and observing the named test
fail; R1.6 by running the amended DoD command against the built store path;
R1.8-R1.16 by reading the code and docs.

- R1.1 - sabotage (collapse the `connect` branch) fails
  `test_open_waits_out_a_concurrent_first_wal_conversion` with "database is
  locked"; the `assert released.is_set()` delivery guard is real.
- R1.2 - sabotage (remove the pre-check) fails
  `test_a_database_at_head_does_not_take_the_write_lock`. PUSHBACK ACCEPTED, and
  reproduced independently: against a raw `sqlite3` holder on `BEGIN IMMEDIATE`,
  `current_revision(db)` returned in 0.000s while the reviewer's suggested
  `engine.connect()` + `_current_revision(conn)` raised "database is locked"
  after 5.006s. The `begin` event fires on the first execute of any SQLAlchemy
  `Connection`, so there is no read-only path through the boundary.
- R1.3 - `os.umask(0o022)` fails the proof at 0o644.
- R1.4 - deleting the call from `main()` fails
  `test_main_configures_logging_and_runs`.
- R1.5 - bypassing `_known_revision` restores the original `CommandError`.
- R1.6 - the amended DoD command passes against the built store; all four
  environment files ship. See R2.2 on the in-repo test clause.
- R1.7 - stubbing `migrate_state_dir` to a no-op fails both tests.
- R1.8, R1.9, R1.10, R1.11, R1.12, R1.13, R1.14, R1.15, R1.16 - confirmed by
  reading the code, docs and `alembic.ini`; the symlink refusal in R1.10 was also
  probed directly.

New findings, all from the round-1 fixes:

- [x] R2.1 (MINOR) scufris/db/migrate.py:93 - the raw-DBAPI read introduced for
  R1.2 changed the exception TYPE on the startup path: a database with a good
  header and damaged pages now raises bare `sqlite3.DatabaseError`, where
  `scufris/README.md` and `open_database`'s docstring both promise
  `sqlalchemy.exc.DatabaseError`. `sqlite3.DatabaseError` is not a subclass of
  it, so a caller written to the live doc would not catch it. The substantive
  invariant - damage never reads as empty - still holds.
  - Response: Fixed. Both doc sentences now state the guarantee accurately (an
    exception, never an empty store) and name `sqlite3.DatabaseError` as the type
    common to both paths, saying which path wraps and which does not;
    `current_revision`'s docstring records that leaving the boundary costs the
    wrapper. Pinned by
    `test_a_damaged_database_raises_at_startup_rather_than_reading_as_empty`,
    which corrupts the pages behind a good header and asserts
    `migrate_state_dir` raises. Making `current_revision` swallow
    `sqlite3.DatabaseError` and return `None` fails it. The wrapping was not
    restored: wrapping would mean re-entering the boundary, which is the write
    lock this read exists to avoid.
- [x] R2.2 (NIT) tests/test_db_migrations.py:340 - the `is_relative_to` clause
  cannot fail; `importlib.resources.files` resolves under `scufris/` by
  construction, and the comment claims it discriminates.
  - Response: Fixed. The comment now says the clause cannot fail on its own and
    that DoD command 4 is the proof the files ship. The assertion is kept as a
    statement of the layout.
- [x] R2.3 (NIT) tasks/20260801-120404/REVIEW.md R1.14 - the Response describes
  the README public-surface table inaccurately.
  - Response: Corrected in place, with the correction marked rather than
    silently rewritten.
- [x] R2.4 (NIT) scufris/db/migrate.py:150 - `os.umask` is process-global.
  - Response: Noted at the call site: safe only because migration is a
    single-threaded startup step that runs before anything else creates files.

Verified independently by the primary:

- The round-2 reviewer ran `ruff check .` (clean), `mypy .` (178 files, clean),
  `python -m pytest` and `nix flake check` (all checks passed), and re-ran the
  two failing node ids in the MASTER checkout to confirm for itself that
  `test_project_tasks_endpoint` and `test_read_project_tasks_parses_real_tatr`
  fail identically there.
- The primary re-ran the same gate plus `nix build .#scufris` and DoD command 4
  before handing the round over, and again after the R2.1-R2.4 fixes.

Pending user checks: none. This task carries no `manual:` proof.
