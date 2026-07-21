# Review: B5bc - retire the Agent protocol + move orchestrator sessions

- TASK: 20260721-180208
- BRANCH: feature/unify-orchestrator

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Fresh subagent, no sight of the implementing session. Ran both suites itself
(backend ruff+mypy+pytest all passed; web 168 passed), verified the deadlock fix
against the actual supervisor FIFO-lock implementation, and confirmed the
load-bearing tests (non-streaming chat, fork) actually PASSED (not skipped).

APPROVE with one MINOR and two NITs, all addressed below.

- [x] R1.1 (MINOR) app.py:501-504 / test_app.py - the on_change ->
  `set_orchestrator_session(None)` cross-backend clear had no app-level test
  (the deleted `test_agent_handle_rebuilds_and_carries_session` used to guard the
  old carry behavior). A behavioral guarantee lost its guard.
  - Response: fixed. Renamed `test_patch_agent_config_rebuilds_on_backend_change`
    to `test_patch_agent_config_backend_change_clears_orchestrator_session`: it
    seeds an orchestrator session, PATCHes `agent_backend` mock->app_server
    through the real endpoint, and asserts `orchestrator_session_id()` is None.
- [x] R1.2 (NIT) app.py:489-494 - stale comment referencing the removed `"chat"`
  serialize key and `supervisor.serialized("chat")`.
  - Response: fixed. Reworded to `agent.id` / `ORCHESTRATOR_ID` serialization and
    added the fork self-deadlock caveat.
- [x] R1.3 (NIT) app.py post_chat_reset - the one rerouted endpoint missing the
  `settings.agent_enabled` 503 guard its siblings have (not a regression, but
  inconsistent).
  - Response: fixed. Added the `agent_enabled` guard for consistency.

### Verified by the reviewer (no issue)

- Deadlock fix real and correct against the per-key FIFO lock; new fork launches
  without the outer lock and is safe (sync set-then-launch, 409 guard).
- No `post_chat` TestClient deadlock; `_drain_turn` yields to the background task.
- No `serialized("chat")` / `serialize_key="chat"` remain in scufris/.
- Session-state move correct; fork/`_drain_turn` read session id off the done
  event, avoiding the persist-callback race.
- Retirement complete (DoD grep empty); MockBackend preserves mock behavior; exec
  runners kept and still tested.
- Disabled gate present on all rerouted endpoints; tests are falsifiable.
- No scope creep (only the expected files changed).
