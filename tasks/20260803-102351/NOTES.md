# Notes: Close the round-2 findings from the create_app assembly extraction

## What changes

No runtime behavior. Three residual defects close: the agent-run router's
booby-trap test starts covering `POST /api/agents/{id}/fork` (which it silently
omitted while claiming `/events` was the only exclusion), `scufris/README.md`
stops pointing the Telegram trust boundary at a symbol that no longer exists,
and a dead logger leaves `host_approval_bridge.py`.

## Surfaces

| File | Why |
|-|-|
| `tests/test_agent_run_router.py` (module docstring + `test_the_agent_run_router_reaches_for_nothing`, line 139) | R2.1: drive `/fork`, correct the "one route left out" claim |
| `tests/test_orchestrator_routers.py:343` (`FakeRunService`) | R2.1: add the `fork_seed` the fork route calls |
| `scufris/api/agent_runs.py:416-447` | R2.1: the route under test; read only |
| `scufris/README.md:85` | R2.2: trust-boundary table names `app._build_telegram_approval_ops` |
| `scufris/telegram/wiring.py::build_approval_ops` | R2.2: where it actually lives now |
| `scufris/host_approval_bridge.py:17,26` | R2.3: `import logging` + an unused module logger (0 `logger.` uses confirmed) |

## Data and interfaces

One fake gains one method, matching what `AgentRunService` exposes
(`scufris/orchestrator/runs.py:468`):

```python
def fork_seed(self, agent: AgentRecord, session_id: str | None,
              message_index: int, text: str) -> str
```

`FakeRunService` records the call and returns a canned seed, the same shape as
its other scripted answers. Nothing in production changes signature.

## Sketches

R2.1 (illustrative):

```python
# tests/test_orchestrator_routers.py - FakeRunService
+   def fork_seed(self, agent, session_id, message_index, text) -> str:
+       self.calls.append(("fork_seed", (agent.id, session_id, message_index, text)))
+       return f"seed:{text}"

# tests/test_agent_run_router.py - inside the trap test
+   assert trap_client.post(
+       f"/api/agents/{AGENT_ID}/fork",
+       json={"message_index": 0, "text": "go"},
+   ).status_code == 200
```

Docstring correction: "``/events`` is the one route left out" stays, and stops
being a lie once `/fork` is driven (14 -> 15 of 16 driven).

## Shape

```
POST /api/agents/{id}/fork
   require_agent_async -> 409 if orchestrator -> 422 if empty text
        -> require_agent_project_async
        -> to_thread(runs.fork_seed, ...)        <-- the fake must answer this
        -> launch(runs, reverted, project, seed) <-- FakeRunService.launch already
        -> relay_bus_sse(bus)                        publishes StreamDone + closes,
                                                     so the TestClient does not hang

/chat takes the identical launch + relay tail and is already driven green,
which is why /fork is safe to add and /events (which blocks) is not.
```

## Consequences and open questions

- All three pointers verified against the tree today: `agent_runs.py:416` is the
  fork route, `README.md:85` still says `app._build_telegram_approval_ops`,
  `host_approval_bridge.py` has zero `logger.` uses. No drift.
- `AGENT_ID` must not be the orchestrator id or the route 409s before reaching
  `fork_seed`; `FakeRunService.project` is already a real `_project()`, so the
  422/404 arms are also clear. Confirm at work time rather than assuming.
- The DoD's grep
  (`! grep -rn _build_telegram_approval_ops --include='*.md' --include='*.py' scufris/ tests/ web/`)
  excludes `tasks/`, which is correct - the one other hit is append-only history.
- After this the trap test drives 15 of 16 routes. `/events` stays out because
  it blocks on a live SSE stream; that exclusion is deliberate and stays
  documented rather than being worked around.
- Cheapest of the five open epic tasks and lowest priority (p20); nothing else
  depends on it.
