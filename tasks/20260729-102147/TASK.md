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
- [ ] Implement the persistence core selected in the decision record: schema or
      snapshot format, connection/lock ownership, and a transaction API whose
      boundary covers mutation plus durable commit.
- [ ] Give the core a safe path for both callers: synchronous FastAPI
      thread-pool routes and asynchronous supervisor callbacks, without
      blocking the event loop on long operations.
- [ ] Migrate `scufris/projects.py` onto the core as the pilot store, leaving
      the other JSON stores untouched and working.
- [ ] Add rollback tests: a failed multi-record operation leaves no partial
      durable state.
- [ ] Add pytest fixtures that give each test an isolated store, replacing any
      ad-hoc temp-directory patterns the pilot exposed.
- [ ] Record the core's public API in the nearest README so the follow-up
      migrations do not re-derive it.

## Definition of Done

- A concurrent burst against the pilot store loses nothing and survives app
  reconstruction (test: `test_concurrent_state_mutations_survive_restart`).
- A failed multi-record operation commits nothing
  (test: `test_state_transaction_rolls_back_as_a_unit`).
- Both a thread-pool route and an asyncio callback can mutate concurrently
  (test: `test_sync_and_async_callers_share_the_transaction_boundary`).
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
