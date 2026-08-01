# Migrate auth, host, schedule, and digest state with a legacy JSON import path

- STATUS: OPEN
- PRIORITY: 78
- TAGS: bug,v0.2.0,reliability,storage,host
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100409

## Story

As a Scufris operator, I want authentication, host proposal, approval,
schedule, and digest state on the same transactional boundary, and a
documented one-shot import of my existing state directory, so that upgrading
never loses a login, a pending approval, or a schedule.

## Steps

- [ ] Write the failing proof first: a host proposal changing state while two
      agents complete, asserting all three survive an app reconstruction.
- [ ] Migrate `scufris/auth/store.py`, `scufris/host_approvals.py`,
      `scufris/scheduler.py`, and `scufris/digest.py` onto the persistence
      core.
- [ ] Keep the root-owned `scufris/hostd/audit.py` log external: reference it,
      do not absorb or rewrite it, and assert the boundary in a test.
- [ ] Land the single entry-point import for a whole legacy state directory:
      backup, validation, partial-migration recovery, and actionable
      diagnostics on corrupt input.
- [ ] Add crash/restart, duplicate-import, corrupt-input, and rollback tests
      over the full state directory.
- [ ] Update `README.md` and `scufris/README.md` with the shipped storage
      model, the migration procedure, backup location, and corruption recovery.
- [ ] Update `examples/` with a runnable proof of the migration if one is
      cheap, or record why not.

## Definition of Done

- Concurrent agent completions and a host proposal change all survive a
  restart (test: `test_concurrent_state_mutations_survive_restart`).
- Authentication, proposal, approval, schedule, and digest fixtures migrate
  transactionally (test: `test_post_host_state_migrates_transactionally`).
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
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the agent-state migration task; this one closes the boundary.
- This task carries the epic's migration documentation and the whole-directory
  import, because it is the first point at which every store is on the core.
- Provide a normal schema-migration path for later conversation/activity
  tables; do not implement those product schemas here.
