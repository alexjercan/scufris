# BC1: durable run-outcome record + AgentState.WAITING (bidirectional comms substrate)

- PRIORITY: 39
- TAGS: spike, agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Add `AgentState.WAITING = "waiting"` to `enums.py` = "ended a turn awaiting
      a decision", distinct from `BLOCKED` (approval) and `DONE`.
- [x] Add an `OutcomeStore` (mirrors the `SessionRegistry` sidecar pattern): a
      per-agent `outcomes.json` under `state_dir`, `agent_id -> RunOutcome
      { state, message, run_id, session_id, ts, acknowledged }`, owned by
      `AgentStore`, atomic write + tolerant load.
- [x] Write the outcome at `mark_finished` (final message threaded from the run's
      `StreamDone.reply.text` via the `persist` callback); `acknowledged=False`,
      `ts=time.time()` on write; for every agent (orchestrator included).
- [x] Expose read accessors on `AgentStore` (`outcome(agent_id)` /
      `outcomes()`).
- [x] `delete(agent_id)` clears the outcome; do not accumulate dead entries.

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

## Close record (2026-07-23)

What changed: added `RunOutcome` (a pydantic model: `state`, `message`,
`run_id`, `session_id`, `ts`, `acknowledged`) and `OutcomeStore` to
`scufris/agent_store.py` - a persisted `<state_dir>/outcomes.json` mapping
`agent_id -> RunOutcome`, atomic write + tolerant load, mirroring
`SessionRegistry` exactly. `AgentStore` owns one (`self._outcomes`), constructed
in `__init__`. `mark_finished` gained `message` + `run_id` params and now writes
a fresh, unacknowledged outcome (`ts=time.time()`) for EVERY agent (orchestrator
included) before the orchestrator/regular split. `delete` clears the outcome
alongside the session mapping. Read accessors `outcome(agent_id)` /
`outcomes()`. In `app.py` `_launch_agent_turn`, the `turn_stream` now captures
`StreamDone.reply.text` into `captured["message"]`, and the `persist` callback
threads `message` + `run_id` into `mark_finished`; `run_id`'s definition moved
above `persist` so the closure captures a clearly-set value. Added
`AgentState.WAITING`. CHANGELOG Added entry.

Evidence: four tests written first, watched fail red for the right reasons
(`AgentState has no attribute 'WAITING'`; `mark_finished() got an unexpected
keyword argument 'message'`; `'AgentStore' object has no attribute 'outcomes'`),
then green: `test_waiting_state_is_distinct`,
`test_run_outcome_persists_and_survives_restart` (the restart leg rebuilds the
store over the same `state_dir`), `test_delete_removes_outcome`,
`test_outcome_store_tolerates_a_corrupt_file`. Full suite 344 passed (340
baseline + 4), `ruff check .` clean, `mypy scufris/agent_store.py app.py
enums.py` clean, from the worktree in the nix dev shell.

Decisions: outcomes recorded for ALL agents (orchestrator included) to keep the
substrate uniform - BC3's `pending_agents` filters to sub-agents rather than the
store deciding. `WAITING` is NOT set here (BC2's `request_input` hard-sets it);
BC1 only names the state and records whatever terminal state `mark_finished`
receives (DONE/ERROR from the completion callback). The final message is stored
whole (uncapped) - one per agent, small; revisit if it ever bloats.

Difficulties: none material. Watched the `run-completion-callback-keys-by-launch-
snapshot` lesson - the outcome's `session_id` is the run's captured id and the
backend keying is unchanged, so no new launch-snapshot hazard was introduced.

Self-reflection: mirroring `SessionRegistry` verbatim made this fast and
low-risk - the sidecar pattern is now well-worn. Threading `message` through
`mark_finished` (rather than writing the outcome directly in the persist
callback) keeps all terminal-run persistence in one method, consistent with how
the session id is already handled.

## Review round 1 follow-up (2026-07-23)

Out-of-context review returned REQUEST_CHANGES on one MAJOR (R1.1). Addressed:
the outcome write was placed ABOVE the regular-agent `_raw` existence check, so
an agent deleted mid-run (the persist callback firing post-delete - an
anticipated path) got a stale outcome resurrected that survived restart,
defeating the delete DoD. The sibling `SessionRegistry.set` sits AFTER `_raw`
and never leaked; the new write was the inconsistency. Fix: build the
`RunOutcome` up front, write it only after existence is established (orchestrator
branch, and after `_raw` for a regular agent). Reproduced red first with
`test_delete_then_mark_finished_does_not_resurrect_outcome` (R1.2), green after.
Also added `test_error_terminal_outcome_recorded` (R1.3). R1.4 (uncapped message)
left as-is for v1. Suite now 346 passed; ruff + mypy clean. Self-reflection: I
mirrored the registry's sidecar SHAPE but not its ORDERING - the existence check
is load-bearing for the delete-race guarantee, and copying a pattern means
copying where its writes sit relative to the guard, not just the class skeleton.
