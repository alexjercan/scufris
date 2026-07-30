# Migrate runtime state to concurrency-safe transactional persistence

- STATUS: OPEN
- PRIORITY: 80
- TAGS: bug,v0.2.0,reliability,storage,backend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a Scufris operator, I want all runtime state to remain correct when requests
and agents finish concurrently, so that successful actions are durable and a
restart never silently drops or corrupts their records.

## Steps

- [ ] Start with failing integration tests for concurrent project mutations,
      simultaneous agent completion callbacks, and restart durability.
- [ ] Implement the persistence layer selected by 20260729-102146 with
      transaction boundaries that cover mutation plus durable commit.
- [ ] Migrate every app-owned mutable store inventoried by 20260729-102146,
      including project, agent, session, outcome, settings, reasoning,
      authentication, and host proposal/schedule state, without changing public
      behavior unnecessarily.
- [ ] Preserve the explicit external boundaries from the persistence decision:
      in particular, reference but do not absorb or rewrite the root-owned
      `scufris-hostd` audit log.
- [ ] Add an idempotent import path for existing JSON state, including backup,
      validation, partial migration recovery, and actionable diagnostics.
- [ ] Ensure synchronous FastAPI thread-pool routes and asynchronous supervisor
      callbacks use the persistence layer safely without blocking the event
      loop on long operations.
- [ ] Add crash/restart, duplicate-import, corrupt-input, and rollback tests.
- [ ] Update configuration, examples, README, and operational recovery
      documentation to match the shipped storage model.

## Definition of Done

- A concurrent burst produces no `500`, lost project, lost session, or lost
  outcome, and all records survive app reconstruction
  (test: `test_concurrent_state_mutations_survive_restart`).
- Existing JSON fixtures migrate exactly once and preserve every supported
  field (test: `test_legacy_json_state_migrates_idempotently`).
- Post-host authentication, proposal, approval, schedule, and digest fixtures
  migrate transactionally, while the privileged audit remains external
  (test: `test_post_host_state_migrates_transactionally`).
- A failed multi-record operation leaves no partial durable state
  (test: `test_state_transaction_rolls_back_as_a_unit`).
- The old fixed shared temporary-file write pattern is absent from runtime
  stores (cmd: `! rg -n 'with_suffix\\(\"\\.json\\.tmp\"\\)' scufris/`).
- The full backend suite remains green (cmd: `python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on: 20260729-102146.
- Starts only after 20260729-124655 has landed; the migration target is the
  post-host state inventory approved in 20260729-102146.
- Preserve the local, single-host deployment model while making in-process
  concurrency correct.
- Prefer one coherent persistence boundary over separate partial fixes in each
  store.
- Provide a normal schema-migration path for later conversation/activity tables;
  do not implement those product schemas here.
