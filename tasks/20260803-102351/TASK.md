# Close the round-2 findings from the create_app assembly extraction

- PRIORITY: 20
- TAGS: refactor, v0.2.0, backend, docs
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a maintainer, I want the three MINOR/NIT findings left open by the router
extraction's APPROVE closed, so that the agent-run trap covers the route it
claims to and `scufris/README.md` names a symbol that exists.

## Steps

- [ ] R2.1a: Add `fork_seed` to `FakeRunService`
      (`tests/test_orchestrator_routers.py:341`), matching
      `AgentRunService.fork_seed` (`scufris/orchestrator/runs.py:468`):
      `def fork_seed(self, agent: AgentRecord, session_id: str | None,
      message_index: int, text: str) -> str`. Record
      `("fork_seed", (agent.id, session_id, message_index, text))` on
      `self.calls` and return a canned seed, the same shape as the neighbouring
      scripted answers.
- [ ] R2.1b: In `test_the_agent_run_router_reaches_for_nothing`
      (`tests/test_agent_run_router.py:139`), after the `/chat` assertion, add
      `assert trap_client.post(f"/api/agents/{AGENT_ID}/fork",
      json={"message_index": 0, "text": "go"}).status_code == 200`. `/fork`
      (`scufris/api/agent_runs.py:416`) does not block the way `/events` does -
      it ends in `launch` + `relay_bus_sse`, exactly like `/chat`, and
      `FakeRunService.launch` already publishes `StreamDone` and closes the bus.
      The trap-test docstring's "``/events`` is the one route left out" needs no
      rewording: driving `/fork` is what makes that sentence true (14 -> 15 of
      16 driven). Reword only if the added assertion leaves it inaccurate.
- [ ] R2.2: `scufris/README.md:85` points the Telegram chat-id allowlist
      re-check at `app._build_telegram_approval_ops`, which task
      20260729-103712 moved to `scufris/telegram/wiring.py::build_approval_ops`.
      Replace the symbol in the trust-boundary table.
- [ ] R2.3: `scufris/host_approval_bridge.py:26` defines
      `logger = logging.getLogger(__name__)` and the module never logs. Delete
      it and the `import logging` on line 17. Both are the file's only
      `logging` tokens.

## Definition of Done

- The agent-run trap drives `/fork` under the same four `__init__` traps, and
  the test still passes
  (cmd: `grep -q 'AGENT_ID}/fork' tests/test_agent_run_router.py && python -m pytest tests/test_agent_run_router.py -k reaches_for_nothing`).
- No source file outside `tasks/` mentions `_build_telegram_approval_ops`
  (cmd: `! grep -rn _build_telegram_approval_ops --include='*.md' --include='*.py' scufris/ tests/ web/`).
- `host_approval_bridge.py` carries no logging machinery
  (cmd: `! grep -n logging scufris/host_approval_bridge.py`).
- The suites pass with no drift
  (cmd: `python -m pytest && ruff check . && mypy .`).

## Notes

- Source: `tasks/20260729-103712/REVIEW.md` round 2, findings R2.1, R2.2, R2.3.
  All three are MINOR or NIT and did not block that task's APPROVE.
- Verified at the time of writing: `scufris/host_approval_bridge.py` has zero
  `logger.` uses; the stale README symbol appears at `scufris/README.md:85` and
  in one `tasks/` record, which is append-only history and exempt.
- All four `cmd:` proofs run red on master before the change (exit 1 each).
  The `/fork` proof needs the `grep -q` guard because the trap test is already
  green on base; the guard is what makes the criterion observe the new
  assertion.
- `AGENT_ID` is `"agent-1"`, not `ORCHESTRATOR_ID`, so `/fork` clears its 409
  arm; `FakeRunService.project` is a real `_project()`, so the 422/404 arms are
  clear too. `text="go"` is non-empty, clearing the 422 empty-text arm.
- After this the trap drives 15 of 16 routes. `/events` stays out on purpose -
  it relays a live bus that nothing closes, so the request never returns.
