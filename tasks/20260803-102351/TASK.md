# Close the round-2 findings from the create_app assembly extraction

- PRIORITY: 20
- TAGS: refactor, v0.2.0, backend, docs
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a maintainer, I want the three MINOR/NIT findings left open by the router
extraction's APPROVE closed, so that the agent-run trap covers the route it
claims to and `scufris/README.md` names a symbol that exists.

## Steps

- [ ] R2.1: `tests/test_agent_run_router.py` drives 14 of the agent-run
      router's 16 routes, but its docstring says "``/events`` is the one route
      left out". `POST /api/agents/{agent_id}/fork`
      (`scufris/api/agent_runs.py:416`) is also undriven and does NOT block the
      way `/events` does - it ends in `launch` + `relay_bus_sse`, exactly like
      `/chat`, which the same test drives green. Add
      `fork_seed(self, agent, session_id, message_index, text) -> str` to
      `FakeRunService` (`tests/test_orchestrator_routers.py:343`) and a
      `trap_client.post(f"/api/agents/{AGENT_ID}/fork", json={"message_index":
      0, "text": "go"})` assertion, then correct the docstring to name only
      `/events`.
- [ ] R2.2: `scufris/README.md:85` points the Telegram chat-id allowlist
      re-check at `app._build_telegram_approval_ops`, which task
      20260729-103712 moved to `scufris/telegram/wiring.py::build_approval_ops`.
      Replace the symbol in the trust-boundary table.
- [ ] R2.3: `scufris/host_approval_bridge.py:26` defines
      `logger = logging.getLogger(__name__)` and the module never logs. Delete
      it and the `import logging` on line 17.

## Definition of Done

- The agent-run trap drives `/fork` under the same four `__init__` traps, and
  its docstring names `/events` as the only exclusion
  (test: `test_the_agent_run_router_reaches_for_nothing`).
- No source file outside `tasks/` mentions `_build_telegram_approval_ops`
  (cmd: `! grep -rn _build_telegram_approval_ops --include='*.md' --include='*.py' scufris/ tests/ web/`).
- The suites pass with no drift
  (cmd: `python -m pytest && ruff check . && mypy .`).

## Notes

- Source: `tasks/20260729-103712/REVIEW.md` round 2, findings R2.1, R2.2, R2.3.
  All three are MINOR or NIT and did not block that task's APPROVE.
- Verified at the time of writing: `scufris/host_approval_bridge.py` has zero
  `logger.` uses; the stale README symbol appears at `scufris/README.md:85` and
  in one `tasks/` record, which is append-only history and exempt.
