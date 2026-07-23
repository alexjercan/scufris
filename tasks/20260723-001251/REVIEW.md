# Review: Persisted agent<->session id registry (fix orchestrator/sub-agent session mixing)

- TASK: 20260723-001251
- BRANCH: fix/session-registry

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and merged

Checked (in-session re-derivation of load-bearing claims): confirmed R1.1 is a
real race by reading the consumers - `update_agent` (app.py:983) updates a
sub-agent's backend with NO `supervisor.serialized(agent_id)` guard, so a switch
can land while a turn is in flight; the finishing `mark_finished` then re-keyed
the session under the current backend. Re-ran the full suite in the worktree
(nix dev shell): `ruff check .` clean, `python -m pytest tests/` 340 passed after
the R1.1 fix, `mypy scufris/agent_store.py scufris/app.py` clean. Sabotage-checked
the new R1.1 test (fails `assert 'codex-sess-late' is None` without the fix) and
re-confirmed the primary repro is red on master.

- [x] R1.1 (MINOR) scufris/agent_store.py:490 - `mark_finished` keyed the
  captured session id by the agent's *current* backend, so a backend switch
  racing an in-flight turn (sub-agent `update_agent` is not serialized against
  turns) would mislabel the finishing session under the new backend and defeat
  the registry's backend-mismatch guard. Suggested: add a `backend` parameter
  fed by the launch-time snapshot.
  - Response: fixed this round. `mark_finished` now takes optional
    `backend`; the supervisor persist callback passes `agent.backend` (the
    launch snapshot). Pinned by
    `test_mark_finished_keys_session_by_run_backend_not_current`,
    sabotage-verified.
- [ ] R1.2 (NIT) scufris/agent_store.py:247 - the legacy migration calls
  `_registry.set` once per record, one file rewrite each. Harmless at real
  agent counts.
  - Response: acknowledged, left as-is (documented for a future touch). One-time
    per-record cost on the first load of a pre-registry file only.

No open `manual:` DoD items for this task (all five DoD proofs are `test:`).
