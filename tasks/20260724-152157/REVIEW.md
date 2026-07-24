# Review: Record the codex session in the registry at turn-start

- TASK: 20260724-152157
- BRANCH: fix/codex-session-at-launch

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

No findings. The out-of-context reviewer diffed against master, read the
surrounding code (`_stream_app_server`, `turn_stream`/`persist`,
`record_running_session`/`mark_finished`/`registry.set`, `session_info` +
`/api/agent/sessions`), and ran the full suite. Confirmed:

- Backend-key agreement (the load-bearing risk): `record_running_session` keys
  under `agent.backend`; the orchestrator record's backend IS `_orch_backend()`
  (`_orchestrator_record`), which is also the read key - so the mid-turn read is
  not None. Same raw-`agent.backend` snapshot `mark_finished` uses.
- Idempotency: both recording paths route through `registry.set`->`add`, which
  dedups the history append; the mid-turn test asserts a single history entry.
- Errored-turn recording: `turn_stream` seeds `captured["session_id"]` on the
  event, so `mark_finished`-on-error still records it (revert-sensitive test).
- Additive frontend + SSE relay: `_relay_bus_sse` serializes the new event
  generically (`model_dump_json`), `dispatchStreamEvent` routes `session_started`
  to an optional `onSessionStarted`, unknown kinds ignored - older consumers safe.
- Edge cases: empty `new_thread_id` guarded; resumed threads re-record the same
  id harmlessly; emitting before `turn/start` is fine (rollout exists at
  thread/start).

Check suite (run in worktree): backend `tests/` all pass (~450); frontend
`vitest` 175/175; `npm run lint` clean; `npm run build` OK.

### In-session supplement (load-bearing re-derivation)

Per the review skill, the in-session pass independently re-derived two claims:

- Record-key vs read-key: confirmed `_orchestrator_record()` sets
  `backend=self._orch_backend()` and `orchestrator_session_id()`/`_sessions()`
  read under `_orch_backend()`, so they agree at launch; a mid-run settings
  backend switch behaves like `mark_finished` (keyed to the launch snapshot) and
  is already covered by `test_patch_agent_config_backend_change_clears_orchestrator_session`.
- Revert-sensitivity: neutering the `record_running_session` call in `turn_stream`
  makes `test_orchestrator_session_recorded_at_turn_start` fail its mid-turn
  assertion (current is None) - the test has teeth at its own boundary.

Open `manual:` DoD item (pending user acceptance, batched at flow Finish):
- On the codex orchestrator chat, send a message and refresh mid-turn -> the
  session shows in the switcher without waiting for the turn to finish.
