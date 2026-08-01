# Cut the project store over to the database

- STATUS: OPEN
- PRIORITY: 80
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120407

## Story

As a Scufris operator, I want the project store to read and write the database
through the transaction boundary, so that the epic's concurrency guarantee is
proven end to end on real state before the remaining stores move.

## Steps

- [ ] Write the epic's headline proof first and watch it fail: a burst of
      concurrent project mutations plus an app reconstruction, asserting no
      `500`, no lost record, and full survival across restart.
- [ ] Call the task-3 importer at startup, ahead of the first store read, so an
      operator's existing `projects.json` is in the database the moment the
      database becomes authoritative.
- [ ] Rewrite `scufris/projects.py` to read through to the database. Delete
      `_projects`, `_load` and `_persist`. No in-memory mirror: 20260729-102146
      measured 97 of 97 failed writes staying live in the process and being
      published by the next successful one.
- [ ] Keep the observable behavior of `create`, `update`, `delete`, `get` and
      `list`, including the `ProjectsReadOnly` gate, the `InvalidProject`
      validations, the slug charset, and `list` ordering by lowercased name.
      The FastAPI routes (`scufris/app.py:1887` onward) and their response
      models do not change.
- [ ] Move `_unique_id` dedup inside the same `db.transaction()` as the insert,
      so the read-modify-write window closes at the transaction rather than at
      the write. Uniqueness is also a real `PRIMARY KEY` now; a collision must
      surface as `DuplicateProject`, not an `IntegrityError` at a route.
- [ ] Keep returning `Project` pydantic instances across the store boundary and
      never let an ORM instance escape - the routes serialize `Project`
      directly, and a `Session`-bound instance in a response is a detached-load
      failure waiting for the first lazy attribute.
- [ ] Grep every `ProjectStore(...)` construction site - `scufris/app.py:946`,
      `scufris/mcp_server.py:81`, and `tests/test_wake.py`,
      `tests/test_agent_sessions.py`, `tests/test_agent_store.py`,
      `tests/test_app.py` - and confirm each reaches a migrated database.
- [ ] Prove the cross-process claim the whole mechanism was chosen for: the MCP
      subprocess and the app see one another's writes. SPIKE.md scenario 5
      measured the JSON alternative losing 150 of 300 cross-process writes with
      `raised=0`.
- [ ] Replace the ad-hoc temp-directory patterns these tests use with the
      task-1 fixture.
- [ ] Update `scufris/README.md` (the project store now reads through the
      database) and add a `CHANGELOG.md` entry.

## Definition of Done

- A concurrent burst against the project store loses nothing and survives app
  reconstruction (test: `test_concurrent_state_mutations_survive_restart`).
- A failed write leaves nothing live in the process
  (test: `test_failed_project_write_leaves_nothing_live_in_memory`).
- An existing `projects.json` is visible through `/api/projects` after the
  upgrade (test: `test_existing_projects_json_is_visible_after_upgrade`).
- The app and an MCP subprocess store see one another's writes
  (test: `test_mcp_and_app_share_one_project_store`).
- A colliding id surfaces as `DuplicateProject`, not a database error
  (test: `test_duplicate_project_id_raises_the_domain_error`).
- The project store no longer uses the fixed shared temporary-file write
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/projects.py`).
- No ORM instance escapes the store - what `list`/`get`/`create`/`update`
  return is a `Project` that still reads its fields after the transaction
  closed (test: `test_project_store_returns_detached_pydantic_records`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145. Lane B, fourth of four. Depends on the legacy import.
- This is the task that closes the epic's Done Means 1 for one store. The
  remaining stores follow in 20260801-100409 and 20260801-100413.
- Lifts the third task's scope fence: wiring the importer's only call site is
  this task's job.
- The other JSON stores stay untouched and working; only `projects.json`
  becomes non-authoritative, and it is copied rather than moved.
- Open question for review, not blocking: `PRAGMA integrity_check` as an
  operator-reachable command (DECISION.md section 4) has no obvious home -
  `scufris/checks.py` is a HOST check registry driven by `HostInspector`.
  Defer to 20260801-100413 unless the core exposes it cheaply.
