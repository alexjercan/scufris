# Migrate auth, host, schedule, and digest state with a legacy JSON import path

- PRIORITY: 78
- TAGS: bug, v0.2.0, reliability, storage, host
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100409

## Story

As a Scufris operator, I want authentication, host proposal, approval,
schedule, and digest state on the same transactional boundary, and a
documented one-shot import of my existing state directory, so that upgrading
never loses a login, a pending approval, or a schedule.

## Steps

Ordered. Steps 1-2 are red-first; every later step keeps the whole suite green.

- [x] **The failing proof, relocated.** Move
      `test_concurrent_state_mutations_survive_restart` out of
      `tests/test_projects.py` (line 277, projects-only today) into a new
      `tests/test_db_state_boundary.py`, and widen it: two agent completions
      and a host proposal changing state concurrently, all three still visible
      after the app is rebuilt from the same state directory. It fails on the
      base because host proposals live in `HostActionStore`'s `OrderedDict`.
      Add `test_post_host_state_uses_declared_persistence_boundary` in the same
      module: every app-owned store constructor in `create_app` takes the
      `Database`, and no runtime store writes a JSON sibling of the state dir.
- [x] **The schema.** Add four rows to `scufris/db/models.py` and autogenerate
      ONE Alembic revision under `scufris/db/migrations/versions/`:
      `AuthSessionRow(id PK, csrf, created_at, last_seen)`,
      `ScheduleRow(name PK, next_due, last_run, last_result, missed, runs)`,
      `DigestRow(id PK autoincrement, at, schedule, verdict, text, delivered,
      delivery_error, states)` and
      `HostActionRow(id PK, seq unique autoincrement, proposal, decision,
      decided_by, decided_at, reason, run_id, result, error)`. `states`,
      `proposal` and `result` are JSON text, as `SettingsOverrideRow.value`
      already is. `test_schema_has_no_pending_autogenerate_diff` is the check.
- [x] **`SessionStore` onto the core** (`scufris/auth/store.py`). Constructor
      takes a `Database`, not a path; `_load`/`_flush`/`self._lock` and the
      in-memory dict go. `prune`, `create`, `get`, `revoke`, `revoke_all` each
      become one `db.transaction()`; `get` keeps its read-renew-expire as a
      single unit of work rather than a read followed by a write. `LoginThrottle`
      stays in memory and unchanged - see DECISION.md 2.
- [x] **Offload the auth call sites.** The auth middleware in `scufris/app.py`
      is `async def`, and `Database.transaction()` refuses a thread with a
      running loop, so `sessions.get` at app.py:1169, :1210, :1322, :1433,
      :1445, :1466 and `sessions.revoke` at :1292, :1308 become
      `await asyncio.to_thread(...)`. `_issue_session` (:1229) is sync and
      called from sync context - confirm each caller, then offload or leave.
      Grep `rg -n "sessions\.(get|create|revoke|prune)" scufris/` and account
      for every hit; a missed one is a 500 on the login path, not a slow path.
- [x] **`SchedulerStore` onto the core** (`scufris/scheduler.py`). `get`,
      `save` and `all` each open one transaction; `get`'s create-on-read
      (scheduler.py:107) is a read-or-insert inside that one transaction rather
      than a read outside it. `HostScheduler.tick`/`run_now`/`_execute` are
      async, so each store call is offloaded with `asyncio.to_thread`. The
      atomic-temp-file write goes, which turns proof 7 green for this file.
- [x] **`DigestStore` onto the core** (`scufris/digest.py`). Add `id: int | None
      = None` to `Digest`, assigned by `add`, so `mark_delivered` updates a row
      by key instead of by object identity; `_load`/`_persist`/the `deque` go.
      The `MAX_DIGESTS` bound becomes a delete of the oldest rows inside the
      insert's transaction. `_run_scheduled_checks` (app.py:2748) is async - the
      four calls at :2756, :2770, :2772, :2777 are offloaded. This turns proof 7
      green for the second file.
- [x] **`HostActionStore` as a decision journal** (`scufris/host_actions.py`),
      per 20260801-100405 DECISION.md section 3. Constructor takes a `Database`;
      `put`, `get`, `list`, `_decide`, `attach_run`, `finish`, `refresh` and
      `_reap` become row operations in one transaction each. `_decide` reads and
      writes inside the SAME transaction, which is what makes `AlreadyDecided`
      hold against two surfaces deciding at once - the current
      read-check-mutate cannot. `deny` writes the reason in that transaction
      too, not as a second mutation afterwards. `HostApprovalService`
      (`scufris/host_approvals.py`) keeps its shape: `refresh_pending` stays
      ADDITIVE, the helper stays authoritative for the PENDING set, and nothing
      here deletes a record the helper stopped listing. Its methods are async,
      so every store call is offloaded.
- [x] **The 0600 assertion.** `FILE_MODE` and the sidecar loop already hold it
      (`scufris/db/engine.py`, `tests/test_db_engine.py:405,427`). Add one test
      in `tests/test_db_state_boundary.py` that logs in through the real app and
      asserts `scufris.db`, `-wal` and `-shm` are all 0600 with a live session
      id in the database.
- [x] **The audit boundary.** Add
      `test_privileged_audit_remains_an_external_boundary`: `scufris/hostd/`
      imports nothing from `scufris.db`, the audit log is still its own
      root-owned file, and an applied action writes BOTH the helper's audit line
      and the app's `host_action` row. Nothing in `scufris/hostd/audit.py`
      changes.
- [x] **One entry point for the whole state directory.** In
      `scufris/db/legacy.py`, add `import_legacy_state(db, state_dir)` as the
      single call `open_state_database` makes, folding in `import_projects` and
      `import_agent_state` plus the three new sources - `auth_sessions.json`,
      `schedules.json`, `digests.json` - under the existing policy: backup at
      0600, validate through the pydantic model, refuse damaged input by name
      with line and column, one transaction and one `legacy_import` gate row per
      source, collect refusals and raise them together. Host actions have no
      legacy file (the store was memory-only); say so in the module docstring
      rather than leaving the absence to be inferred.
- [x] **Whole-directory import tests** in `tests/test_db_legacy.py`: a full
      state directory imports once; a second start is a no-op and duplicates
      nothing; a damaged `schedules.json` fails startup by name while the other
      sources still land with their gate rows; a record that fails validation
      mid-file rolls the whole source back and leaves no gate row.
      `test_post_host_state_migrates_transactionally` and
      `test_host_proposal_decisions_survive_restart` (decision, operator and
      reason survive a restart, while `refresh_pending` still reconciles the
      pending set from the helper) live in `tests/test_db_state_boundary.py`.
- [x] **Docs.** `README.md` "The state directory, backups and downgrade" already
      names the three required facts (`-wal`/`-shm` in a backup, damaged is
      refused by name rather than repaired, downgrade only while the legacy JSON
      exists). What is stale is the sentence "auth sessions, host state, the
      schedule and the digest history are still JSON" - replace it with the
      shipped model: every app-owned store on `scufris.db`, the privileged audit
      log outside it. Add the new sources and the whole-directory import to
      `scufris/README.md` section 9.
- [x] **Example.** Decided: `examples/state_migration.py` is worth having.
      `examples/` is this repo's declared home for runnable end-to-end proofs, the
      claim is operator-facing ("upgrading never loses a login"), and the script
      is the only artifact that shows the whole upgrade in one read: a legacy
      directory, the import, a pre-upgrade cookie still authenticating, a second
      start that changes nothing, and a damaged file refused by name. Exits 0.

## Definition of Done

- Concurrent agent completions and a host proposal change all survive a
  restart (test: `test_concurrent_state_mutations_survive_restart`).
- Authentication, proposal, approval, schedule, and digest fixtures migrate
  transactionally (test: `test_post_host_state_migrates_transactionally`).
- A host proposal's decision, operator and reason survive a restart, while the
  pending set still reconciles from the helper
  (test: `test_host_proposal_decisions_survive_restart`).
- The privileged audit log stays outside the app store
  (test: `test_privileged_audit_remains_an_external_boundary`).
- A whole legacy state directory migrates exactly once
  (test: `test_legacy_json_state_migrates_idempotently`).
- Every app-owned store shares the declared boundary
  (test: `test_post_host_state_uses_declared_persistence_boundary`).
- The fixed shared temporary-file write is gone from runtime stores
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/`).
- Migration and recovery are documented
  (cmd: `rg -n "migration|backup|recovery" README.md scufris/README.md`).
- The README no longer describes auth, host, schedule or digest state as JSON
  (cmd: `! rg -n "are still JSON" README.md`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Close-out

**What and why.** Every app-owned store is now on `Database.transaction()`, and
an operator's whole legacy state directory is read in through ONE call.
`import_legacy_state(db, state_dir)` replaced `import_projects` and
`import_agent_state`, added `auth_sessions.json`, `schedules.json` and
`digests.json` under the existing policy, and states in the module docstring that
host actions have no legacy source because the store was memory-only.
`scufris/db/legacy.py` became the package `scufris/db/legacy/` - `gate.py` (backup,
gate row, refusal, `import_legacy_file`) and `loaders.py` (one loader per source) -
when the added loaders pushed it past the repo's 600-line source cap.

**Alternatives.** Keeping the two old entry points beside the new one was
rejected: with the sources split, a damaged `projects.json` raised before the
agent sources were attempted, so the "one refusal does not hold back the other
sources" property held within each half and not across the directory. Recorded as
DECISION.md 5, with the example decision (Step 13) taken FOR the example rather
than against it.

**Difficulties and diagnosis.** Two fixture corrections, both the same mistake:
`test_post_host_state_migrates_transactionally` was written with 1970 timestamps,
so the imported session was correctly pruned at startup and the schedule's past
`next_due` was correctly counted as a missed window - the test was measuring the
sweep and the tick rather than the import. Live timestamps make it test what it
claims. `SchedulerState` no longer exists (Step 5 deleted the whole-file model),
so the schedules loader validates per entry through `ScheduleState`, keyed by the
mapping key rather than the copy of the name on the record.

**Review round 1.** Two findings, both fixed on the branch.

R1.1 was a permanent unbootable startup, reached through the repair path this
task documents. Collecting refusals means a later source still runs when an
earlier one was refused, and for one pair that changes what the later source
writes: refuse `sessions.json`, and `load_agents` migrates each pre-registry
`session_id` into `agent_session` itself and gates `agents.json`. The repaired
file then arrived for agents that had rows, and `load_sessions`' plain insert
raised `UNIQUE constraint failed: agent_session.agent_id` - uncaught, and
unclearable, because the agents gate row means the conflicting write is never
replayed. `load_sessions` now upserts the mapping and replaces the agent's
history rows, which is also the correct rule: that file is the switcher's own
record and the id on an agent record was only ever the stand-in for it. The
defect predates this branch - master's `import_agent_state` has the same loop,
ordering and "degrades correctly" sentence - but this branch rewrote that
docstring, moved it to the new single entry point and widened the policy across
the whole directory, so it was fixed here rather than deferred. DECISION.md 6.

R1.2 said the boundary test listed its six stores by hand and so could not fail
for a store added later. Rewritten to DISCOVER them off `app.state`, and the
discovery immediately falsified this task's own "every app-owned store" claim:
`ConfigChangeStore` is still an in-memory `OrderedDict`. Migrating a fifth store
was materially outside these Steps, so it became 20260803-002141 under the same
epic, and the test excludes `config_changes` by name against that ID while
asserting the exclusion is still needed.

**Evidence.** `ruff check .`, `ruff format --check .` (191 files) and `mypy .`
(191 files) clean; the full suite exits 0. Every DoD proof run individually
(proofs 1-6 as six named tests, 7 and 9 as the negated greps, 8 as the doc grep,
10 as the whole check line), plus
`test_repairing_a_refused_sessions_file_completes_the_import`, which fails on the
parent commit with the exact IntegrityError it now prevents.
`tatr check 20260801-100413` clean; `python examples/state_migration.py` exits 0.

**Reflection.** The file-size cap is what forced the package split, and the split
is better than the file was - the policy prose, the mechanism and the per-source
loaders were three things in one module. Worth reaching for earlier next time: the
cap caught it, but only after the last loader landed.

The review's two findings share one root, and it is worth naming: both were
places where a record ASSERTED a property that nothing executed. "Degrades
correctly" was prose about a path no test walked, and "every app-owned store"
was a claim a hand-written list could not falsify. Each became true only once
something ran it - a repair-and-restart test, and a discovery walk. Where a
docstring claims a recovery path, that path is a test; where it quantifies over
a set, the test derives the set rather than repeating it.

## Notes

- Epic: 20260729-102145.
- Depends on the agent-state migration task; this one closes the boundary.
- This task carries the epic's migration documentation and the whole-directory
  import, because it is the first point at which every store is on the core.
- Provide a normal schema-migration path for later conversation/activity
  tables; do not implement those product schemas here.
- Discovered while planning: `Database.transaction()` refuses a thread with a
  running event loop (`scufris/db/engine.py`), and the auth middleware, the
  scheduler tick and every `HostApprovalService` method are `async def`. The
  offloads are the largest and riskiest part of this task, not the schema.
- `test_concurrent_state_mutations_survive_restart` exists today at
  `tests/test_projects.py:277` and covers projects only. It moves rather than
  being duplicated - see DECISION.md 1.
- `HostActionStore` has no legacy JSON file to import: the spike found it
  memory-only, rebuilt from the root helper.
- `LoginThrottle` is out of scope as durable state - DECISION.md 2.
- The three facts the decision requires in writing are ALREADY in `README.md`
  ("The state directory, backups and downgrade"); what this task owes is
  correcting the stale "still JSON" inventory, not writing them again.
