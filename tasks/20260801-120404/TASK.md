# Land the Alembic migration runner and the projects schema

- PRIORITY: 82
- TAGS: bug, v0.2.0, reliability, storage, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102147

## Story

As a Scufris maintainer, I want schema evolution to be reviewable, generated
and drift-tested before any store depends on it, so that the two follow-up
store migrations add a revision instead of inventing a migration mechanism.

## Steps

- [x] Write the failing proofs first: a fresh state dir reaching head, a second
      startup that is a no-op, and an autogenerate comparison that reports no
      diff between the declarative models and the migrated database.
- [x] Add `alembic` with `uv add`, `uv lock`, re-enter `nix develop`, and
      confirm the uv2nix closure still builds. Alembic pulls `Mako`.
- [x] Define the schema declaratively in `scufris/db/models.py` with SQLAlchemy
      2.0 `DeclarativeBase` + `Mapped[...]`. One table, `projects`, matching the
      `Project` fields at `scufris/projects.py:55` with `id` as the primary key.
      Nothing else - no agent, session, outcome, auth or host tables, and no
      conversation, activity-event or delivery tables.
- [x] Put the Alembic environment at `scufris/db/migrations/` (`env.py`,
      `script.py.mako`, `versions/`), NOT at the repo root: the wheel is built
      with `only-include = ["scufris"]` (`pyproject.toml`), so a root
      `alembic/` directory would not ship to an operator.
- [x] Configure Alembic programmatically at startup - build an
      `alembic.config.Config`, set `script_location` from
      `importlib.resources.files("scufris.db.migrations")` and NO
      `sqlalchemy.url`, then `command.upgrade(cfg, "head")`. Reuse the task-1
      engine rather than letting `env.py` open a second one, so the production
      pragmas apply to the migration connection too.
      AMENDED in review round 1 (process signal): the original clause said to set
      "the URL from `Settings.state_dir`". Setting it is exactly what would let
      `env.py` fall back to opening its own engine on SQLite's defaults, which
      contradicts the Step's own next clause. The connection is handed over on
      `cfg.attributes` instead.
- [x] Keep a repo-root `alembic.ini` for `alembic revision --autogenerate`
      during development only, and say in `scufris/README.md` that it is a dev
      tool and not the runtime path.
- [x] Write the first revision creating `projects`. Run `ruff format` on it -
      the Alembic template does not satisfy the repo's Ruff config, and this is
      the step every later revision repeats.
- [x] Wire `upgrade head` into app construction ahead of any store, and check
      it also runs for the MCP subprocess entry point
      (`scufris/mcp_server.py:81`), which opens the same database.
- [x] Extend the task-1 pytest fixture to apply `upgrade head`, so every later
      test gets a migrated database.
- [x] Document the revision workflow in `scufris/README.md`: autogenerate,
      review, `ruff format`, and the drift test that catches a forgotten one.

## Definition of Done

- A fresh state dir reaches head and a second startup changes nothing
  (test: `test_migrations_reach_head_and_are_idempotent`).
- The declarative models and the migrated schema agree - autogenerate reports
  no diff (test: `test_schema_has_no_pending_autogenerate_diff`).
- App construction migrates the database before any store reads it
  (test: `test_app_startup_upgrades_state_schema`).
- The migration scripts ship inside the wheel
  (cmd: `nix build .#scufris && test -f "$(dirname "$(readlink -f result/bin/scufris)")/../lib/python3.14/site-packages/scufris/db/migrations/script.py.mako"`).
  AMENDED in review round 1 (R1.6): the original form ran
  `importlib.resources.files('scufris.db.migrations')` in-process, which inside
  the repo resolves to the WORKTREE and so passes even if the wheel excluded the
  files. It has to read the BUILT output to prove anything.
- The migration connection carries the production pragmas, not defaults
  (test: `test_migration_connection_uses_production_pragmas`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).
- The Nix closure still builds with the new dependency
  (cmd: `nix build .#scufris`).

## Notes

- Epic: 20260729-102145. Lane B, second of four. Depends on the engine core.
- The `projects` table is created here but nothing reads or writes it yet:
  `scufris/projects.py` is still on JSON until the fourth task. That is
  deliberate and is what lets this land green.
- Alembic replaces the `PRAGMA user_version` ladder of 20260801-100405
  DECISION.md section 4. Everything else in that section - backup before
  migration, never delete a legacy file, validate through pydantic, refuse
  damaged input - is unchanged and lands in the next task.
- `VACUUM INTO '<state_dir>/scufris.db.pre-v<N>.bak'` before a schema migration
  (DECISION.md section 4) belongs here, in the runner, not in each revision.
- Sync Alembic only. An async env would need `aiosqlite`, which DECISION.md
  rejected on measurement.

## Close-out

### What and why

Alembic 1.18.5 (pulling `Mako` 1.3.12) is in `pyproject.toml` and `uv.lock`, and
`scufris/db/` gained three things: `models.py` (the declarative schema, one
table), `migrations/` (the Alembic environment, shipped INSIDE the package) and
`migrate.py` (the runner). `migrate_state_dir(state_dir)` is the startup call;
`create_app` makes it ahead of `ProjectStore`, and `mcp_server.main` makes it for
the orchestrator subprocess, which opens the same file.

Nothing reads or writes `projects` yet - `ProjectStore` is still on
`projects.json` - which is what lets this land green on its own.

Two properties are load-bearing and each has a proof that FAILS without it:

- **The migration runs on the app's own engine.** `migrate.py` opens a
  connection from `Database.engine`, begins it, and hands it to `env.py` on
  `config.attributes`. So the schema change inherits WAL, `synchronous=FULL`,
  the busy timeout and foreign keys instead of SQLite's defaults. No
  `sqlalchemy.url` is set on that path, so an `env.py` that fell back to
  dialling its own engine fails loudly rather than migrating the right file with
  the wrong pragmas.
- **The revision check and the upgrade share one `BEGIN IMMEDIATE`.** The
  dashboard and an MCP subprocess can start together; the second waits on the
  write lock, then reads a revision that is already head and does nothing. A
  check outside the lock would have both decide to create the same table.
  REVISED in round 1 (R1.2): a lock-free raw read now runs FIRST and returns
  early, so the common startup - already at head - never takes the lock at all.
  The locked re-read above is unchanged and still owns the check-then-act.

The pre-migration backup (`VACUUM INTO scufris.db.pre-<revision>.bak`, mode
0600) is in the runner as the Notes required, taken only when the database is
behind head AND has been migrated before - a fresh database has nothing to
protect.

### Alternatives considered

- **`Config` with `sqlalchemy.url` from `Settings.state_dir`, as Step 5 spelled
  it.** The same Step also requires reusing the task-1 engine rather than letting
  `env.py` open a second one, and the two cannot both hold: a URL is exactly how
  `env.py` opens its own. The connection is handed over on `config.attributes`
  instead and the URL is deliberately absent, which is the same Step's stated
  intent and is what makes the pragma proof discriminate. The URL form survives
  only in the root `alembic.ini`, which the runtime never reads.
- **A repo-root `alembic/` directory.** Rejected in the plan and confirmed here:
  the wheel is `only-include = ["scufris"]`, and the built package was checked
  from outside the repo to be sure `env.py`, `script.py.mako` and the revision
  all ship.
- **Migrating inside the shared `database` fixture only.** Done, per Step 9 - but
  it means `test_db_migrations.py` cannot use that fixture for anything, since a
  runner cannot be shown to reach head from a database that starts there. It has
  its own `fresh` fixture that opens an unmigrated database.
- **An integration test of the backup through `upgrade_to_head`.** Not reachable:
  the branch needs a database BEHIND head, and this build has one revision and it
  is the first. `backup_database` is proven directly instead - a complete,
  `integrity_check`-clean copy at 0600 - plus two tests that no backup is taken
  on a fresh database or on one already at head. The wiring gets its first real
  exercise at the next revision.
- **Offline mode (`alembic upgrade --sql`).** Refused with a message rather than
  half-supported: this schema is applied in place to a local SQLite file by the
  process that owns it, so there is no reviewer to hand a script to.

### Difficulties and diagnosis

`test_migration_connection_uses_production_pragmas` PASSED under its own
sabotage on the first attempt. The test captured every statement on the engine
and read pragmas off the first one - but the runner reads the current revision on
that engine before it migrates, so an `env.py` that then went off and dialled its
own engine still left a captured connection behind. Fixed by capturing only the
connection that executed `CREATE TABLE projects`, i.e. the DDL itself; the
sabotage then failed with "the schema was not created on the app's own engine".
This is the second time on this lane that the first version of a proof restated
the code instead of discriminating against its failure mode.

`test_app_startup_upgrades_state_schema` had the same weakness - asserting on the
finished app would pass with the migration wired in AFTER every store. It now
wraps `ProjectStore` and records the revision at construction time; moving
`migrate_state_dir` one line later fails it with `[None] != ['8f8087f3cc9c']`.

The generated revision needed `ruff check --fix` as well as `ruff format` (one
blank line in the import block). Step 7 named only the format pass, so the
documented workflow in `scufris/README.md` names both.

### Evidence

All seven named proofs pass. Falsified rather than merely observed green:

| Sabotage | What broke |
|-|-|
| dropped `description` from the revision | `test_schema_has_no_pending_autogenerate_diff` (an `add_column` diff) and `test_projects_table_matches_the_project_record` |
| `env.py` ignores the handed-over connection and dials its own engine from a URL | `test_migration_connection_uses_production_pragmas`: "the schema was not created on the app's own engine" |
| `migrate_state_dir` moved after `ProjectStore(settings)` | `test_app_startup_upgrades_state_schema`: `[None] != ['8f8087f3cc9c']` |

- `ruff check . && mypy .`: clean (178 source files).
- `python -m pytest`: 922 passed, 2 failed - `test_project_tasks_endpoint` and
  `test_read_project_tasks_parses_real_tatr`, both shelling out to a real `tatr`
  and both failing identically on `master`. Pre-existing, unrelated, and skipped
  under `nix flake check`.
- `nix flake check`: all checks passed. `nix build .#scufris`: builds.
- The wheel proof was run from OUTSIDE the repo against the built environment,
  so it could not pass on the source tree by accident: `env.py`,
  `script.py.mako` and `8f8087f3cc9c_create_projects.py` all ship.
- The documented dev loop was run end to end: `alembic upgrade head` then
  `alembic revision --autogenerate` against the scratch database produced an
  empty revision, which is the drift check by hand.

### Reflection

Both proofs that mattered were wrong on the first write, in the same way: they
asserted the good outcome rather than the mechanism that produces it, and only
the sabotage said so. Writing the sabotage BEFORE calling a proof done - not
after - would have caught both in one pass instead of two.

The `VACUUM INTO` backup is the one piece of this task shipped without an
end-to-end exercise, purely because a one-revision history cannot express "behind
head". The next revision should assert it on the real path rather than trusting
the seam test.

No CHANGELOG entry existed for the engine core because nothing was operator
visible; this task IS the first to put a file in an operator's state directory,
so it adds one that says plainly that nothing else has changed for them.

## Close-out: review round 1 (REQUEST_CHANGES)

Sixteen findings, all addressed; Responses are on each one in `REVIEW.md`. Four
changed behavior rather than wording.

### What changed and why

- **The fresh-database WAL race (R1.1).** `PRAGMA journal_mode=WAL` returns
  SQLITE_BUSY WITHOUT invoking the busy handler, so `busy_timeout` never covered
  the one-time delete->WAL conversion: two processes on a fresh state dir raced
  and one died at open, before any lock of its own. `engine._set_journal_mode`
  is the busy handler SQLite declines to run - a bounded retry over the same 5s.
- **The startup pre-check (R1.2).** Every begin on this engine is
  `BEGIN IMMEDIATE`, so the old code took the write lock merely to learn there
  was nothing to do, and `busy_timeout` turns a wait over 5s into a hard failure.
  `current_revision` now reads on a raw DBAPI connection - the reviewer's
  suggested `engine.connect()` still takes the lock, which the new test proved -
  and the locked re-read stays for the check-then-act.
- **The backup's permissions and its guards (R1.3, R1.5, R1.10).** `VACUUM INTO`
  runs under `umask(0o077)` so the copy is 0600 from creation rather than
  narrowed after; a revision this build does not know is refused BEFORE any
  backup is written; a symlinked target is refused like everywhere else.
- **The MCP entry point (R1.4).** The one-line `migrate_state` wrapper is gone
  and the assertion moved onto the real call site in
  `test_main_configures_logging_and_runs`.

### Difficulties and diagnosis

- The reviewer's own suggested fix for R1.2 does not work on this engine, and
  only the test said so: the plain-connection pre-check failed "database is
  locked". That is the boundary doing what it was designed to do - one begin,
  always immediate - so the read had to leave SQLAlchemy rather than the boundary
  gain a read-only mode.
- `test_a_database_at_head_does_not_take_the_write_lock` passed its first
  sabotage for the wrong reason: two `Database` objects on one path in one
  process trip the re-entrancy guard, not the lock. Holding the lock from a raw
  `sqlite3` connection reproduces cross-process contention faithfully.
- The first attempt at the 0600 proof watched the file mid-`VACUUM` through
  `set_progress_handler`; on a database this small it never fired. Monkeypatching
  `os.chmod` to a no-op and asserting the CREATED mode is deterministic.
- mypy rejects `**MIGRATION_CONTEXT_OPTS` into `context.configure` (11 errors):
  the values still come from the one constant, but they are passed by key.

### Evidence

Each round-2 fix is pinned by a proof that fails without it:

| Sabotage | What broke |
|-|-|
| remove the `_set_journal_mode` branch from the connect hook | `test_open_waits_out_a_concurrent_first_wal_conversion` |
| `os.umask(0o022)` around `VACUUM INTO` | `test_the_backup_is_never_world_readable_even_briefly` (0o644) |
| remove the lock-free pre-check | `test_a_database_at_head_does_not_take_the_write_lock` ("database is locked") |
| skip the `_known_revision` check | `test_a_database_from_a_newer_scufris_is_refused_without_a_backup` |
| remove `migrate_state_dir` from `mcp_server.main` | `test_main_configures_logging_and_runs` |

Full re-verification, not just the tests near the fixes: `ruff check .` clean,
`mypy .` clean on 178 files, `python -m pytest` 2 failed (the same two
pre-existing `tatr`-shelling tests), `nix flake check` all checks passed,
`nix build .#scufris` builds, and the amended DoD proof 4 finds
`script.py.mako` under the built store path.

### Reflection

The sabotage moved into the test-writing step this round, as the review's third
process signal asked. It caught two bad proofs immediately instead of at
verification - and one of those, R1.2's, also falsified the reviewer's suggested
fix. A proof written to discriminate is worth more than the finding that
prompted it.
