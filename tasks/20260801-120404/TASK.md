# Land the Alembic migration runner and the projects schema

- STATUS: OPEN
- PRIORITY: 82
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102147

## Story

As a Scufris maintainer, I want schema evolution to be reviewable, generated
and drift-tested before any store depends on it, so that the two follow-up
store migrations add a revision instead of inventing a migration mechanism.

## Steps

- [ ] Write the failing proofs first: a fresh state dir reaching head, a second
      startup that is a no-op, and an autogenerate comparison that reports no
      diff between the declarative models and the migrated database.
- [ ] Add `alembic` with `uv add`, `uv lock`, re-enter `nix develop`, and
      confirm the uv2nix closure still builds. Alembic pulls `Mako`.
- [ ] Define the schema declaratively in `scufris/db/models.py` with SQLAlchemy
      2.0 `DeclarativeBase` + `Mapped[...]`. One table, `projects`, matching the
      `Project` fields at `scufris/projects.py:55` with `id` as the primary key.
      Nothing else - no agent, session, outcome, auth or host tables, and no
      conversation, activity-event or delivery tables.
- [ ] Put the Alembic environment at `scufris/db/migrations/` (`env.py`,
      `script.py.mako`, `versions/`), NOT at the repo root: the wheel is built
      with `only-include = ["scufris"]` (`pyproject.toml`), so a root
      `alembic/` directory would not ship to an operator.
- [ ] Configure Alembic programmatically at startup - build an
      `alembic.config.Config`, set `script_location` from
      `importlib.resources.files("scufris.db.migrations")` and the URL from
      `Settings.state_dir`, then `command.upgrade(cfg, "head")`. Reuse the
      task-1 engine rather than letting `env.py` open a second one, so the
      production pragmas apply to the migration connection too.
- [ ] Keep a repo-root `alembic.ini` for `alembic revision --autogenerate`
      during development only, and say in `scufris/README.md` that it is a dev
      tool and not the runtime path.
- [ ] Write the first revision creating `projects`. Run `ruff format` on it -
      the Alembic template does not satisfy the repo's Ruff config, and this is
      the step every later revision repeats.
- [ ] Wire `upgrade head` into app construction ahead of any store, and check
      it also runs for the MCP subprocess entry point
      (`scufris/mcp_server.py:81`), which opens the same database.
- [ ] Extend the task-1 pytest fixture to apply `upgrade head`, so every later
      test gets a migrated database.
- [ ] Document the revision workflow in `scufris/README.md`: autogenerate,
      review, `ruff format`, and the drift test that catches a forgotten one.

## Definition of Done

- A fresh state dir reaches head and a second startup changes nothing
  (test: `test_migrations_reach_head_and_are_idempotent`).
- The declarative models and the migrated schema agree - autogenerate reports
  no diff (test: `test_schema_has_no_pending_autogenerate_diff`).
- App construction migrates the database before any store reads it
  (test: `test_app_startup_upgrades_state_schema`).
- The migration scripts ship inside the wheel
  (cmd: `python -c "import importlib.resources as r; assert (r.files('scufris.db.migrations') / 'env.py').is_file()"`).
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
