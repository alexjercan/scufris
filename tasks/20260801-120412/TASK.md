# Cut the project store over to the database

- STATUS: CLOSED
- PRIORITY: 80
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120407

## Story

As a Scufris operator, I want the project store to read and write the database
through the transaction boundary, so that the epic's concurrency guarantee is
proven end to end on real state before the remaining stores move.

## Steps

- [x] Write the epic's headline proof first and watch it fail: a burst of
      concurrent project mutations plus an app reconstruction, asserting no
      `500`, no lost record, and full survival across restart.
- [x] Call the task-3 importer at startup, ahead of the first store read, so an
      operator's existing `projects.json` is in the database the moment the
      database becomes authoritative.
- [x] Rewrite `scufris/projects.py` to read through to the database. Delete
      `_projects`, `_load` and `_persist`. No in-memory mirror: 20260729-102146
      measured 97 of 97 failed writes staying live in the process and being
      published by the next successful one.
- [x] Keep the observable behavior of `create`, `update`, `delete`, `get` and
      `list`, including the `ProjectsReadOnly` gate, the `InvalidProject`
      validations, the slug charset, and `list` ordering by lowercased name.
      The FastAPI routes (`scufris/app.py:1887` onward) and their response
      models do not change.
- [x] Move `_unique_id` dedup inside the same `db.transaction()` as the insert,
      so the read-modify-write window closes at the transaction rather than at
      the write. Uniqueness is also a real `PRIMARY KEY` now; a collision must
      surface as `DuplicateProject`, not an `IntegrityError` at a route.
- [x] Keep returning `Project` pydantic instances across the store boundary and
      never let an ORM instance escape - the routes serialize `Project`
      directly, and a `Session`-bound instance in a response is a detached-load
      failure waiting for the first lazy attribute.
- [x] Grep every `ProjectStore(...)` construction site - `scufris/app.py:946`,
      `scufris/mcp_server.py:81`, and `tests/test_wake.py`,
      `tests/test_agent_sessions.py`, `tests/test_agent_store.py`,
      `tests/test_app.py` - and confirm each reaches a migrated database.
- [x] Prove the cross-process claim the whole mechanism was chosen for: the MCP
      subprocess and the app see one another's writes. SPIKE.md scenario 5
      measured the JSON alternative losing 150 of 300 cross-process writes with
      `raised=0`.
- [x] Replace the ad-hoc temp-directory patterns these tests use with the
      task-1 fixture.
- [x] Update `scufris/README.md` (the project store now reads through the
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

## Close-out

### What and why

`ProjectStore` now reads and writes the `projects` table through
`Database.transaction()`. `_projects`, `_load` and `_persist` are gone: there is
no in-memory mirror, so a write that fails leaves nothing behind and cannot be
published by the next successful one. `_unique_id` moved inside the same
transaction as the insert, and the id is a real `PRIMARY KEY`, so the
read-modify-write window closes at the transaction rather than at the write;
the constraint's violation is mapped to `DuplicateProject` so no
`IntegrityError` can surface at a route. Nothing but a pydantic `Project`
crosses the boundary - the statements run on the Core `Connection`, so there is
no ORM instance that could escape into a response.

Startup grew one entry point, `scufris.db.open_state_database`: open, migrate,
import legacy JSON, hand the caller the long-lived handle. `create_app` uses it
and closes the handle in the lifespan; the MCP subprocesses reach it through the
new `scufris/mcp_stores.py`, which memoizes one `Database` per resolved state
dir. That module also absorbed the store wiring that used to sit in
`mcp_server.py`, which was 16 lines over the 600-line file cap after this
change.

Observable behavior is unchanged: the routes, the response models, the
`ProjectsReadOnly` gate, the `InvalidProject` validations, the slug charset, and
`list` ordering by lowercased name all stayed put.

### Alternatives considered

- **`ProjectStore(settings)` opening its own database.** Rejected: it keeps 39
  call sites untouched but gives every store its own engine and hides who owns
  the handle's lifetime. The epic's premise is ONE boundary per process, so the
  handle is constructed once and injected.
- **Ordering `list` in SQL.** Rejected: SQLite's `lower()` is ASCII-only, and
  the ordering the API has always published is Python's `str.lower`.
- **Keeping `migrate_state_dir`.** Deleted instead: `open_state_database`
  replaced its only two production callers, and a public export nothing calls
  invites the next store to wire up the wrong one.

### Difficulties and diagnosis

- **Import cycle.** `scufris.projects` needs `Database` and `ProjectRow`;
  `scufris.db.legacy` needs `Project`. Importing `scufris.projects` first would
  reach `legacy`'s top-level `from ..projects import Project` before `Project`
  existed. Fixed by deferring that one import into `_load_projects`, with the
  reason recorded there.
- **A pre-existing red test.** `test_read_project_tasks_parses_real_tatr` failed
  on the base commit: `tatr ls` now emits `KIND` and `FLOW STEP` between
  `PRIORITY` and `TAGS`, and `_TASK_LINE_RE` pinned the exact field list, so
  every task silently vanished from the Projects page. It gated this task's
  `python -m pytest` proof and lives in the file being rewritten, so it was
  fixed here: the regex now skips whatever sits between the two fields it reads.
- **The 20260729-102146 repro harness.** Its projects scenario drove the store
  that no longer exists. Rewired onto the database rather than deleted, so it
  now measures the replacement: 200 expected, 200 recovered after a restart, 0
  raised. The other three scenarios still reproduce their failures, as they
  should - those stores have not moved.

### Evidence

- `ruff check .` clean, `mypy .` clean on 182 files, `python -m pytest` 941
  passed.
- All seven named proofs pass; the `.json.tmp` grep returns nothing.
- Independent of pytest: `python tasks/20260729-102146/repro_state_races.py`
  reports `projects (state database)` at expected 200, after_restart 200,
  create_raised 0, exceptions 0. The same harness measured 103 of 200 recovered
  and 97 raised-but-live against the JSON store.
- `test_mcp_and_app_share_one_project_store` runs a real child process, so the
  cross-process claim is proven across a process boundary rather than simulated.

### Reflection

The plan's "replace the ad-hoc temp-directory patterns with the task-1 fixture"
was the largest part of the work by line count and the smallest by risk - about
40 mechanical call-site edits across eight test files. Doing it with `sed` and
then reading every diff was faster than editing by hand and no less careful, but
it also reformatted five unrelated files that were never `ruff format`-clean on
master; those were reverted so the diff stays on-topic. Worth knowing before the
next two store cutovers, which will touch the same test files again.

### Review round 1

Four findings, all addressed; see REVIEW.md for the per-finding responses.

The one that mattered was the MAJOR: three `async def` agent routes resolved an
agent's project by calling the store directly, so a `BEGIN IMMEDIATE`
transaction ran on the event loop thread and took SQLite's single write lock
there. The review measured the loop stalling 3.04s behind a held lock against a
0.01s heartbeat, and past `busy_timeout` the wait becomes a 500. Missed here
because the cutover was reviewed as a store change - the store's own callers
were checked, and `_require_agent_project`, one indirection away, was not. The
lesson generalizes to the next two cutovers: moving a store onto the database
changes every caller's cost model, so the sweep is "who calls this, from which
thread", not "who constructs this".

Verification after the fixes: `ruff check .` clean, `mypy .` clean on 182 files,
`python -m pytest` 943 passed, all seven named proofs green.
