# Review: BC4 auto-wake bridge

- TASK: 20260723-094313
- BRANCH: feat/wake-bridge

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

The out-of-context reviewer ran the full suite (377 passed, ruff + mypy clean),
ran each DoD proof by name, and did a deep concurrency audit confirming all the
load-bearing properties BY CONSTRUCTION: (1) self-deadlock SAFE - in
`supervisor.py` `_execute` the `finally` calls `release()` BEFORE `on_complete`,
so when the finishing run (sub-agent OR the orchestrator itself) wakes the
orchestrator, its serialize key is already freed; no path holds `ORCHESTRATOR_ID`
at launch. (2) drain-when-idle CORRECT - `run.state=DONE` is set before the
`finally`, so the orchestrator's own completion sees `_orchestrator_busy()` False.
(3) reentrancy SAFE - `supervisor.start` schedules via `create_task`, so
`on_complete` cannot re-enter `on_run_complete` synchronously; `_drain` has no
await, so `_pending` is race-free. (4) 409-absorb - `_pending` is popped per-id
only on launch success, so a wake is never dropped. It independently
SABOTAGE-verified the integration test (stubbing the `on_run_complete` call makes
`test_auto_wake_launches_orchestrator_on_subagent_waiting` fail "orchestrator was
not woken", prompts=['do it'], then restored). In session I had confirmed the same
completion-callback ordering by reading the supervisor before designing.

- [ ] R1.1 (NIT) scufris/wake.py `on_run_complete` - re-writes `_pending[agent_id]`
  on every completion for a still-pending agent. Harmless (a fresh completion
  carries the current message); noted for awareness.
  - Response: acknowledged - it is in fact desired (a new completion should
    refresh the message); a still-pending agent only re-enters via a genuinely new
    completion of its own run. Left as-is.
- [ ] R1.2 (NIT) tests/test_app.py OFF integration test - asserts orchestrator not
  running after a fixed 0.1s sleep. Sound because auto_wake off is a pure no-op
  (nothing is ever scheduled), so it cannot flake into a false pass.
  - Response: left as-is (the reviewer confirmed it cannot false-pass).

No open `manual:` DoD items (all proofs are `test:`/`cmd:`).
