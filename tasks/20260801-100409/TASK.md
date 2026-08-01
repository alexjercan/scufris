# Migrate agent, session, outcome, settings, and reasoning state

- STATUS: OPEN
- PRIORITY: 79
- TAGS: bug, v0.2.0, reliability, storage, agents
- KIND: TASK
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120412

## Story

As a Scufris operator, I want agent, session, outcome, settings, and reasoning
state on the transactional core, so that simultaneous agent completions never
drop a session record or an outcome that the UI already reported.

## Steps

- [ ] Write the failing proof first: simultaneous agent completion callbacks
      writing registry, outcome, and reasoning records, asserting every record
      survives an app reconstruction.
- [ ] Migrate `scufris/agent_store/store.py`, `registry.py`, and `outcomes.py`
      onto the persistence core, keeping the public store APIs stable and
      reading through to the database instead of mirroring rows in memory.
- [ ] Migrate `scufris/settings_store.py`, and replace the reasoning sidecar's
      per-session JSON files with `(session_id, seq)` rows: 20260801-100405
      measured snapshot append cost rising with history, and this is the store
      that already loses turns silently.
- [ ] Close the read-modify-write windows 20260729-102146 listed for these
      stores inside the transaction, not around the write: `mark_finished`'s
      `preserve_signal` read, `OutcomeStore.acknowledge`, `SessionRegistry.add`
      and `remove`, and `SettingsStore.apply`.
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
- Reasoning turns are rows, not per-session files, and a turn is never lost
  silently (test: `test_reasoning_turns_persist_without_swallowing_errors`).
- No agent-state store uses the fixed shared temporary-file write
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/agent_store/ scufris/settings_store.py scufris/reasoning_store.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145.
- Depends on the persistence core task; that task owns the transaction API.
- Auth, host, schedule, and digest state migrate in the successor task; they
  keep working on JSON until then.
- The reasoning sidecar's error swallowing (`reasoning_store.py:120`) is
  removed, not ported: it is why 186 of 200 turns disappeared with no failed
  request in 20260729-102146.
