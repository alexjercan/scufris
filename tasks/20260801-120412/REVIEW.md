# Review: Cut the project store over to the database

- TASK: 20260801-120412
- BRANCH: fix/project-store-database

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/app.py:2869 - `_require_agent_project` calls
  `projects.get()`, which is now a synchronous `BEGIN IMMEDIATE` transaction,
  directly on the event loop thread from three `async def` routes
  (`run_agent` at :2869, `agent_chat` at :3006, `agent_fork` at :3109).
  `scufris/db/engine.py` states the rule this breaks: "Loop-thread callers
  therefore wrap a SYNCHRONOUS unit of work and offload it:
  `await asyncio.to_thread(unit_of_work)`", and because every begin is
  immediate, even this read takes the single write lock. Measured on this
  branch: with another connection holding the write lock, `store.get()` on the
  loop blocked for 3.04s and the loop's worst heartbeat gap was 3.04s against a
  0.01s target - every other request, SSE stream and health probe in the
  process is stalled for the wait, and past `busy_timeout=5000` the route turns
  a lock wait into an `OperationalError` 500. Change the three async call sites
  to `project = await asyncio.to_thread(_require_agent_project, agent)`; the
  sync caller at :3362 can keep calling it directly.
  - Response: fixed - `_require_agent_project_async` (scufris/app.py:2218) wraps
    the sync helper in `asyncio.to_thread`, and `run_agent`, `agent_chat` and
    `agent_fork` await it; `get_agent_capabilities` still calls the sync one
    directly. Pinned by `test_project_lookup_never_runs_on_the_event_loop`
    (tests/test_app.py:2897), which drives all three routes and asserts the
    thread that opens the transaction has no running loop - the invariant
    itself, not a timing. Red before the fix with exactly three loop-thread
    lookups.
- [x] R1.2 (MINOR) scufris/projects.py:216 - `update` now validates `name` and
  `cwd` before it looks the project up, where the old store did
  `project = self.get(project_id)` first. `PATCH /api/projects/ghost` with an
  empty name or a missing `cwd` therefore returns 422 where it returned 404,
  against the Step's "keep the observable behavior of ... `update`". Restore
  the old precedence: resolve the record first (a `_fetch` before the
  validations, or the validations moved after the `_fetch` inside the
  transaction with the `is_dir` call left outside it), and pin it with a case
  in `test_get_and_delete_unknown_raise`.
  - Response: fixed - the validations moved inside the transaction, after the
    `_fetch` (scufris/projects.py:212). The `is_dir` stat is now held under the
    write lock, which `create` avoids; that is deliberate and commented - one
    stat costs far less than the second immediate begin an existence probe
    ahead of the validations would. `test_get_and_delete_unknown_raise` covers
    the empty name and the missing cwd against an unknown id.
- [x] R1.3 (MINOR) tests/test_projects.py:1 - `test_store_ignores_corrupt_file`
  was deleted and nothing replaced it at the boundary where the behavior
  actually changed. Refusing a damaged legacy file is tested at the import
  layer (`tests/test_db_legacy.py:104`), but the operator-facing promise this
  diff creates - README.md:163 and CHANGELOG.md:80, "the startup fails rather
  than presenting you with an empty store" - is untested. Verified by hand that
  it holds: `create_app` with a damaged `projects.json` raises
  `LegacyImportRefused`. Add that as a test beside the other app-level proofs
  in this file.
  - Response: fixed - `test_a_damaged_projects_json_stops_the_app_starting`
    (tests/test_projects.py:339) asserts `create_app` raises
    `LegacyImportRefused` naming `projects.json` and the position it stops
    parsing at, which is what README.md and the changelog promise.
- [x] R1.4 (MINOR) CHANGELOG.md:44 - the `_TASK_LINE_RE` fix
  (scufris/projects.py:32) repairs an operator-visible bug - every tatr task
  silently vanished from the Projects page once `tatr ls` grew `KIND` and
  `FLOW STEP` - and gets no `### Fixed` entry, while smaller operator-visible
  fixes in this release do. Add one naming the symptom and the cause.
  - Response: fixed - a `### Fixed` entry at CHANGELOG.md:60 names the symptom
    (no tasks on the Projects page), the cause (`tatr ls` grew `KIND` and
    `FLOW STEP` between the two fields the parser pinned) and the change.

Verified independently:

- `ruff check .` clean, `mypy .` clean on 182 source files, `python -m pytest`
  exit 0. `ruff format --check` reports 5 files, all of them unformatted on
  `master` too (6 there); this branch formats one of them and adds none.
- `python tasks/20260729-102146/repro_state_races.py` re-run from the branch:
  `projects (state database)` reports expected 200, in_memory 200, on_disk 200,
  after_restart 200, create_raised 0, raised_but_live_in_memory 0. The
  close-out's number is real. The other three scenarios still fail, as their
  stores have not moved.
- The `.json.tmp` grep proof returns nothing.
- `create_app` on a good legacy `projects.json` leaves
  `projects.json.pre-sqlite.bak` beside the untouched original, as README.md
  and CHANGELOG.md claim.
- Read every store method against the diff: dedup and insert share one
  transaction, `delete` decides on `rowcount` inside it, `update` re-reads
  inside it, `list` buffers `Row`s and converts after commit, and the only type
  crossing the boundary is `Project`. `_unique_id`'s `autoescape=True` is
  needed, not defensive - a slug cannot contain `_`, but an id imported from
  legacy JSON can.

Process signal: the branch also fixes an unrelated pre-existing failure
(`test_read_project_tasks_parses_real_tatr`, the `_TASK_LINE_RE` regex). It sat
in the file being rewritten and gated this task's own `python -m pytest` proof,
so fixing it here is defensible, but it is a second user-visible bug landing
under a storage-cutover task with no record of its own.

Process signal: the 600-line file cap forced `scufris/mcp_stores.py` out of
`mcp_server.py` mid-task. The next two store cutovers touch the same MCP wiring
and the same eight test files; the plan naming that extraction up front would
keep it out of the cutover diffs.

## Round 2

- REVIEWER: in-session (verification of four recorded fixes on an unchanged
  design; this session's policy forbids dispatching a subagent reviewer)
- VERDICT: APPROVE

All four round-1 findings verified fixed; no fix regressions found. What was
re-derived rather than read:

- R1.1: the round-1 test is red on the pre-fix tree with exactly three
  loop-thread lookups (`[False, True, True, True]`) and green after, and the
  offload is the only thing that changed between them. `test_agent_fork_validates`
  still gets 422 for an orphaned project, so `HTTPException` raised inside the
  worker thread still reaches the route unchanged.
- R1.2: checked at the HTTP boundary, not just the store - `PATCH
  /api/projects/ghost` with an empty name and with a missing `cwd` both return
  404 again, matching `master`. Moving `is_dir` under the write lock is the
  right trade at one stat against a second immediate begin, and it is
  commented.
- R1.3: `test_a_damaged_projects_json_stops_the_app_starting` cannot pass
  without the startup wiring - `pytest.raises` fails if `create_app` returns.
- R1.4: the `### Fixed` entry names symptom, cause and change.

Full suite re-run on the fix commit: `ruff check .` clean, `mypy .` clean on
182 source files, `python -m pytest` 943 passed (up 2 from round 1's 941), all
seven named proofs green, the `.json.tmp` grep still empty.

No pending `manual:` proofs on this task.
