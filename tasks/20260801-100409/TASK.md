# Migrate agent, session, outcome, settings, and reasoning state

- STATUS: OPEN
- PRIORITY: 79
- TAGS: bug,v0.2.0,reliability,storage,agents
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102147

## Story

As a Scufris operator, I want agent, session, outcome, settings, and reasoning
state on the transactional core, so that simultaneous agent completions never
drop a session record or an outcome that the UI already reported.

## Steps

- [ ] Write the failing proof first: simultaneous agent completion callbacks
      writing registry, outcome, and reasoning records, asserting every record
      survives an app reconstruction.
- [ ] Migrate `scufris/agent_store/store.py`, `registry.py`, and `outcomes.py`
      onto the persistence core, keeping the public store APIs stable.
- [ ] Migrate `scufris/settings_store.py` and `scufris/reasoning_store.py`.
- [ ] Commit the multi-record completion path (registry update plus outcome
      append plus reasoning append) inside a single transaction.
- [ ] Add an idempotent import for the legacy JSON files these stores own,
      with backup and validation, reusing the migration policy from the
      decision record.
- [ ] Add duplicate-import and corrupt-input tests for these stores.
- [ ] Update `scufris/README.md` where the agent-state storage description
      drifts from what ships.

## Definition of Done

- Simultaneous completions lose no session, outcome, or reasoning record
  (test: `test_concurrent_agent_completions_persist_every_record`).
- The completion path is atomic across its records
  (test: `test_agent_completion_commits_as_one_transaction`).
- Legacy agent-state JSON fixtures import exactly once and preserve every
  supported field (test: `test_legacy_agent_state_migrates_idempotently`).
- No agent-state store uses the fixed shared temporary-file write
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/agent_store/ scufris/settings_store.py scufris/reasoning_store.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the persistence core task; that task owns the transaction API.
- Auth, host, schedule, and digest state migrate in the successor task; they
  keep working on JSON until then.
