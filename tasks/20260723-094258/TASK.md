# BC1: durable run-outcome record + AgentState.WAITING (bidirectional comms substrate)

- STATUS: OPEN
- PRIORITY: 39
- TAGS: spike,agents,backend

## Story

As the orchestrator, I want every sub-agent run to leave a DURABLE outcome
(final message + terminal state) that outlives the ephemeral per-run EventBus, so
I can observe a finished agent later instead of holding a blocking `message_agent`
call. This is the substrate the rest of the bidirectional-comms work builds on.

## Context (grounded)

The per-run EventBus is ephemeral: `supervisor.start` creates one per run
(`supervisor.py:181-219`) and `_execute` calls `bus.close()` on completion
(`supervisor.py:250-293`); `GET /api/agents/{id}/events` (`app.py:1228-1242`)
404s when no run is active. So a signal that must outlive a run cannot live on
the bus. Run completion IS already captured durably: the on-complete callback
(`app.py:1112-1132`) calls `agents.mark_finished(id, state=, session_id=,
backend=)` (`agent_store.py:467-504`), which persists terminal state + writes the
session to the `SessionRegistry` sidecar (`sessions.json`,
`agent_store.py:103-167`). This task extends that seam to also record the final
message. `AgentState` (`enums.py:47-54`) has no "ended awaiting a decision"
value - `BLOCKED` means "waiting on an approval".

Spike: `tasks/20260723-001256/SPIKE.md` (BC1).

## Steps (/plan expands)

- [ ] Add `AgentState.WAITING = "waiting"` to `enums.py` = "ended a turn awaiting
      a decision", distinct from `BLOCKED` (approval) and `DONE`.
- [ ] Add an `OutcomeStore` (or reuse the `SessionRegistry` sidecar pattern): a
      per-agent `outcomes.json` under `state_dir`, `agent_id -> { run_id,
      session_id, state, message, ts, acknowledged }`, owned by `AgentStore`,
      atomic write + tolerant load (mirror `agent_store.py:103-167`).
- [ ] Write the outcome at `mark_finished` (final message from the run's
      `StreamDone.reply` / `read_status.last_message`); `acknowledged=false` on
      write.
- [ ] Expose a read accessor on `AgentStore` (`outcome(agent_id)` /
      `outcomes()`), keyed by the record's current backend where relevant.
- [ ] `delete(agent_id)` clears the outcome; do not accumulate dead entries.

## Definition of Done

- After a faked sub-agent run ends, its outcome (final message + terminal state)
  is readable from a freshly rebuilt `AgentStore` over the same `state_dir`
  (survives a simulated restart).
  (test: `test_run_outcome_persists_and_survives_restart`)
- `AgentState.WAITING` exists and is distinct from `BLOCKED`/`DONE`.
  (test: covered above)
- Deleting an agent removes its outcome entry.
  (test: `test_delete_removes_outcome`)
- `ruff check .`, `mypy` on touched files, and `python -m pytest` are green from
  the worktree. (cmd: `python -m pytest`)

## Notes

- Mirror the just-landed `SessionRegistry` (`tasks/20260723-001251`) for the
  sidecar shape - same atomic-write/tolerant-load discipline.
- Lessons: `persist-callback-must-not-raise` (the outcome write must not throw in
  the on-complete callback), `mark_finished-keys-by-launch-snapshot-backend`.
- Spike-seeded (BC1); depends on nothing. BC2/BC3/BC4 depend on this.
