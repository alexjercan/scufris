# Introduce the transactional persistence core and migrate the project store

- STATUS: OPEN
- PRIORITY: 80
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100405

## Story

As a Scufris operator, I want the persistence mechanism chosen by the spike to
exist as a working transaction boundary with one store on top of it, so that
the concurrency guarantee is proven end to end before the remaining stores
move.

## Steps

- [ ] Write the failing concurrency proof first: a burst of concurrent project
      mutations plus an app reconstruction, asserting no `500`, no lost record,
      and full survival across restart.
- [ ] Implement the core from 20260801-100405 DECISION.md section 1-2: one
      SQLite database at `<state_dir>/scufris.db` through stdlib `sqlite3`,
      the four pragmas at 0600, a connection per thread, and a single
      synchronous `db.transaction()` context manager over `BEGIN IMMEDIATE`.
- [ ] Land the `PRAGMA user_version` migration runner with the core, not with
      the last store: an ordered `(version, callable)` list where each entry
      runs inside one transaction that also bumps the version.
- [ ] Give the core a safe path for both callers without a second async API:
      loop-thread callers offload a synchronous unit of work through
      `asyncio.to_thread`. Enforce that a transaction never spans an `await`,
      since coroutines on the loop thread share one connection.
- [ ] Migrate `scufris/projects.py` onto the core as the pilot store, reading
      through to the database rather than mirroring rows in memory, and leaving
      the other JSON stores untouched and working.
- [ ] Import an existing `projects.json` under the same policy the whole-directory
      import will use (backup, validation, idempotent, refuse damaged input), so
      the pilot does not strand an operator's projects for two tasks.
- [ ] Add rollback tests: a failed multi-record operation leaves no partial
      durable state, and a failed mutation leaves nothing live in memory either.
- [ ] Add pytest fixtures giving each test a file-backed database under
      `tmp_path` with the production pragmas, replacing any ad-hoc
      temp-directory patterns the pilot exposed.
- [ ] Record the core's public API and its rules in `scufris/README.md` so the
      follow-up migrations do not re-derive them.

## Definition of Done

- A concurrent burst against the pilot store loses nothing and survives app
  reconstruction (test: `test_concurrent_state_mutations_survive_restart`).
- A failed multi-record operation commits nothing
  (test: `test_state_transaction_rolls_back_as_a_unit`).
- Both a thread-pool route and an asyncio callback can mutate concurrently
  (test: `test_sync_and_async_callers_share_the_transaction_boundary`).
- An existing `projects.json` imports once, keeps a backup, and a damaged one
  is refused with its location named
  (test: `test_legacy_projects_import_is_idempotent_and_refuses_damage`).
- A damaged store raises instead of presenting itself as empty
  (test: `test_damaged_state_refuses_to_load`).
- The project store no longer uses the fixed shared temporary-file write
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/projects.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the persistence decision spike; the mechanism is not re-litigated
  here.
- Pilot-store scope is deliberate: the other stores keep their current JSON
  files and behavior, so this lands green on its own.
- Preserve the local single-host deployment model; the goal is in-process
  concurrency correctness, not a networked database.
- Read-through rather than an in-memory mirror is a decision constraint, not a
  style preference: 20260729-102146 measured 97 of 97 failed writes staying
  live in the process and being published by the next successful one.
- Create no conversation, activity-event or delivery tables. The chosen store
  carries them later through a normal `user_version` migration; SPIKE.md is the
  proof, 20260729-220835 designs them.
