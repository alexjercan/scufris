# Review: Session ownership index + multi-session history; drive the switcher from it

- TASK: 20260724-111947
- BRANCH: fix/session-ownership-index

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (run in the worktree by the out-of-context reviewer and re-run
in-session): `pytest` green (172 passed in the store+app suites), `ruff check`
clean, `mypy scufris` clean. Every DoD proof executed and passing: the five
named tests pass, and `grep -n list_sessions scufris/app.py` prints nothing
(switcher no longer disk-scans). The leak repro's red-on-master was confirmed by
inspection (master's `get_sessions` filters only by `(originator, cwd)`, so both
`orch-sess` and `sub-sess` codex_exec rollouts in the same cwd survive the
filter and fail the `== ["orch-sess"]` assertion).

Independently re-derived in-session (not adopted from the reviewer wholesale):
the orchestrator backend-switch path (`_update_orchestrator` ->
`set_orchestrator_session(None)`) resolves `_orch_backend()` to the NEW backend,
so `_entry` mismatches the old entry and `_fresh` starts an empty history under
the new backend - old codex ids do not resurface; and `set_current(None)` under
a matching backend preserves `sessions` while nulling `session_id` (new chat
keeps history). Both hold.

- [x] R1.1 (NIT) tests/test_app.py:1451 (`test_orchestrator_switcher_lists_registry_history`)
  and scufris/app.py:1664 - the DoD names this test as proving "newest first,"
  but it asserts only set membership (`{"sess-old","sess-new"}`), and both
  fixtures share an identical mtime, so the `_activity` sort key + `reverse=True`
  are effectively untested (a strict order assert would be flaky as written).
  Suggest writing the two rollouts with distinct mtimes and asserting list
  order, so the "newest first" claim is actually covered. Sort logic is correct
  by inspection; coverage gap, not a bug.
  - Response: fixed - the test now `os.utime`s the two rollouts to distinct
    mtimes (1000 vs 2000) and asserts the list order is `["sess-new",
    "sess-old"]`. A/B confirmed it goes red without `reverse=True` and green
    with it, so the sort is now genuinely covered. Verified in-session; ticked
    on that confirmation.

No BLOCKER/MAJOR/MINOR findings. No open `manual:` DoD items (all proofs are
`test:`/`cmd:`). Verdict APPROVE; the NIT was addressed as a discretionary
follow-up. Cycle closed.
