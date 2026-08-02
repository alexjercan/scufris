# Migrate auth, host, schedule, and digest state with a legacy JSON import path

- PRIORITY: 78
- TAGS: bug, v0.2.0, reliability, storage, host
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
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
      core. `SchedulerStore.get` writes on a read path
      (`scufris/scheduler.py:107`); that write belongs inside a transaction.
- [ ] Give `HostActionStore` a durable decision journal per 20260801-100405
      DECISION.md section 3: the decision, operator, reason and apply result
      persist, while `HostApprovalService.refresh_pending` keeps its additive
      semantics and the root helper stays authoritative for the PENDING set.
- [ ] Keep the database at mode 0600 once auth session identifiers live in it,
      matching what `scufris/auth/store.py` protects today.
- [ ] Keep the root-owned `scufris/hostd/audit.py` log external: reference it,
      do not absorb or rewrite it, and assert the boundary in a test.
- [ ] Land the single entry-point import for a whole legacy state directory:
      backup, validation, partial-migration recovery, and actionable
      diagnostics on corrupt input.
- [ ] Add crash/restart, duplicate-import, corrupt-input, and rollback tests
      over the full state directory.
- [ ] Update `README.md` and `scufris/README.md` with the shipped storage
      model, the migration procedure, backup location, and corruption recovery.
      Name three things the decision requires in writing: that a backup must
      include `-wal` and `-shm`, that a damaged store is refused with a named
      remedy rather than repaired, and that downgrading works only while the
      legacy JSON files are still present.
- [ ] Update `examples/` with a runnable proof of the migration if one is
      cheap, or record why not.

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
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the agent-state migration task; this one closes the boundary.
- This task carries the epic's migration documentation and the whole-directory
  import, because it is the first point at which every store is on the core.
- Provide a normal schema-migration path for later conversation/activity
  tables; do not implement those product schemas here.
