# Notes: record codex session at turn-start

## Verify-first findings

- **Rollout is written at thread/start.** Inspected a real
  `~/.codex/sessions/2026/07/24/rollout-*.jsonl`: its FIRST event is
  `type: session_meta` (before any `user_message`/`agent_message`). So a
  just-started codex thread is already a readable on-disk session mid-turn. This
  underpins two things: (1) `session_info` (`scufris/backends.py:1017`) returns
  None only when `not messages and status is None`, and `read_status` reads the
  rollout via `_find_rollout` which finds the file from thread/start, so the
  in-flight session is NOT dropped from `/api/agent/sessions` - it lists as
  "(untitled)" until the first user message flushes; (2) the failed-turn case
  (thread started, turn errored) is a real session, so recording ownership on it
  is correct.

## What changed

- `StreamSessionStarted(kind="session_started", session_id)` added to the
  `StreamEvent` union (`scufris/agent.py`), emitted in `_stream_app_server` right
  after `new_thread_id` is set (before `turn/start`), for both fresh and resumed
  threads.
- `AgentStore.record_running_session(agent_id, backend, session_id)`
  (`scufris/agent_store.py`) - thin wrapper over `registry.set`, keyed under the
  launch-time backend, idempotent with `mark_finished`.
- `_launch_agent_turn`'s `turn_stream` (`scufris/app.py`) records ownership on the
  event and seeds `captured["session_id"]` so an errored turn still persists it.
- Frontend: `StreamSessionStartedEvent` in `web/src/common.ts`; routed in
  `web/src/chat-stream.ts` `dispatchStreamEvent` to an optional `onSessionStarted`
  (no UI yet - the landing consumes it in task 20260724-152230).

## Tests

- `test_orchestrator_session_recorded_at_turn_start` - drives a blocking mock
  stream that emits `StreamSessionStarted` first; asserts
  `orchestrator_session_id()` is set WHILE the run is live (before release/done),
  persists after settle, and history has a single entry (no double-append).
- `test_orchestrator_session_recorded_even_when_turn_errors` - StreamSessionStarted
  then StreamError (no done): the session is still recorded.
- `dispatchStreamEvent` unit: a `session_started` frame routes to
  `onSessionStarted`, never to `onError`.

See DECISION.md for the mechanism choice (event vs callback vs status-poll).
