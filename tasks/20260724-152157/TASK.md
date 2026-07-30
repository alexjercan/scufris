# Record codex session in the registry at turn-start (early StreamSessionStarted -> set_current)

- STATUS: CLOSED
- PRIORITY: 85
- TAGS: bug, agents, sessions, backend, codex
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As someone using the orchestrator chat on the codex backend, when I send a
message and refresh the page mid-turn, I want the session to already exist in
the switcher, so that I do not have to wait for the whole turn to finish before
the conversation shows up.

Part of umbrella 20260724-151911. Scope: codex only.

## Root cause (investigated)

scufris records a run's session id in its registry only in the terminal
`persist` callback -> `agents.mark_finished(... session_id=...)`
(`scufris/app.py:1199-1217`, `scufris/agent_store.py:705`). `mark_running`
(`agent_store.py:641`) records no session. So mid-turn
`orchestrator_session_id()` (`agent_store.py:682`) is still the pre-turn value
(None for a fresh chat) and `/api/agent/sessions` shows nothing for it.

The codex thread/session id is known EARLY: in `_stream_app_server`
(`scufris/agent.py`), after `thread/start`/`thread/resume` returns,
`new_thread_id = thread["id"]` at agent.py:573-575 - BEFORE `turn/start` and the
streaming loop. It is only surfaced today at the end via
`StreamDone(..., session_id=new_thread_id)` (agent.py:644).

## Approach

Surface the thread id the moment it is known via a new additive StreamEvent
kind, and record ownership in the registry immediately from the run-launch path
(instead of waiting for `mark_finished`). Codex-only: other backends need not
emit it (additive; unknown kinds ignored by `dispatchStreamEvent`). See the
DECISION.md in this task folder for the mechanism choice (new event vs callback).

## Steps

- [x] Verify-first (rollout timing): inspect a real `~/.codex/.../rollout-*.jsonl`
      to confirm codex writes `session_meta` (the rollout file) at thread/start, so
      a just-started thread is a readable on-disk session mid-turn. Record the
      finding in NOTES.md. (Grounds steps below + the failed-turn case.)
- [x] Add a `StreamSessionStarted` StreamEvent (`kind="session_started"`,
      `session_id: str`) to the union in `scufris/agent.py:63-94`.
- [x] Emit it in `_stream_app_server` right after `new_thread_id` is set
      (`agent.py:573-575`), before `turn/start`, whenever `new_thread_id` is truthy
      (covers both `thread/start` fresh and `thread/resume`).
- [x] In `_launch_agent_turn`'s `turn_stream` (`scufris/app.py:1177-1195`), on a
      `StreamSessionStarted` event record ownership immediately: for the
      orchestrator `agents.set_orchestrator_session(ev.session_id)`; for a sub-agent
      the registry `set_current(agent.id, agent.backend, ev.session_id)` path. Key
      it under the LAUNCH-TIME backend snapshot (`agent.backend`), same reasoning as
      `mark_finished`'s `backend=` param, so a mid-run backend switch cannot
      mislabel it. Yield the event through unchanged.
- [x] Confirm idempotency: `set_current` sets the current pointer AND appends to
      history if new (`agent_store.py:237`), so the later `mark_finished` recording
      the same id is a no-op re-set. Add/keep a test asserting no double history
      entry.
- [x] Confirm `/api/agent/sessions` (`scufris/app.py:~1710-1731`) surfaces the
      in-flight session: `session_info` (`scufris/backends.py:1017`) returns None
      only when `not messages and status is None`; a just-started codex rollout has
      a readable status (rollout on disk from thread/start), so it lists as
      "(untitled)". Add an endpoint test that the current session appears mid-turn.
- [x] Frontend type: add the `StreamSessionStarted` shape to `web/src/common.ts`
      StreamEvent union and route it in `web/src/chat-stream.ts` `dispatchStreamEvent`
      to an optional `onSessionStarted?(id)` handler (additive; no UI change here -
      the landing consumes it in the follow-up task). Keep it a no-op default so
      existing callers are unaffected.
- [x] Backend test: drive a codex-shaped stream that yields `StreamSessionStarted`
      early then blocks before `StreamDone`; assert `orchestrator_session_id()` /
      `/api/agent/sessions` `current` is set WHILE the run is live (mirror the
      blocking-stream harness in `tests/test_app.py` around
      `test_agent_chat_conflicts_with_active_run` and the Q1-A test
      `test_status_exposes_in_flight_prompt_stripped`).
- [x] Failed-turn test: a stream that yields `StreamSessionStarted` then
      `StreamError` (no `StreamDone`) still leaves the session recorded (consistent
      with `mark_finished`-on-error), and does not corrupt the registry.
- [x] Write DECISION.md for the early-session-event mechanism; index it in the
      umbrella GOAL.md Decisions section.
- [x] Run the full gate: `nix flake check` and `cd web && npm run ci`; both green.

## Definition of Done

- On a fresh codex orchestrator turn, the session id is in the registry as
  current at turn-start, before the terminal frame (test: new `tests/test_app.py`
  case asserting `/api/agent/sessions` `current` mid-turn).
- `/api/agent/sessions` lists the in-flight session mid-turn (test: same/adjacent
  case asserting it appears in `sessions`).
- A thread-started-then-errored turn still records the session and leaves the
  registry consistent (test: the failed-turn case).
- Recording at turn-start does not double-write history vs the terminal
  `mark_finished` (test: history has one entry for the id).
- Full gate green (cmd: `nix flake check`) and web green (cmd: `cd web && npm run ci`).
- manual: on the codex orchestrator chat, send a message and refresh mid-turn ->
  the session shows in the switcher without waiting for the turn to finish.

## Notes

- Key files: `scufris/agent.py` (StreamEvent union :63-94, `_stream_app_server`
  thread id :573-575, StreamDone :644), `scufris/app.py` (`_launch_agent_turn`
  :1153-1239, `/api/agent/sessions` :~1710-1731), `scufris/agent_store.py`
  (`set_orchestrator_session` :661, `set_current` :237, `mark_finished` :705),
  `scufris/backends.py` (`session_info` :1017), `web/src/common.ts` (StreamEvent),
  `web/src/chat-stream.ts` (`dispatchStreamEvent`).
- Composes with the just-landed Q1-A change (`AgentRunStatus.prompt` +
  `onUserPrompt`): once the landing reflects the live turn (follow-up task), the
  refreshed page shows session + prompt + streaming reply.
- The mock backend will not emit `StreamSessionStarted`; the backend test drives a
  stream that yields it (a fake stream or monkeypatched MockBackend.stream), not
  the real codex binary.
- Assumption to confirm in step 1: codex writes the rollout at thread/start (the
  rollouts on disk begin with a `session_meta` event before any `user_message`).

## Close-out (what changed, why, difficulties, reflection)

Implemented as planned; see DECISION.md (mechanism) and NOTES.md (verify-first
findings + change list). Summary:

- `StreamSessionStarted(kind="session_started", session_id)` added to the
  `StreamEvent` union, emitted in `_stream_app_server` right after `new_thread_id`
  is set (before `turn/start`, for fresh + resumed threads).
- `AgentStore.record_running_session(agent_id, backend, session_id)` records
  ownership under the launch-time backend; `turn_stream` calls it on the event and
  seeds `captured["session_id"]` so an errored turn still persists the id.
- Frontend: `StreamSessionStartedEvent` in common.ts, routed in
  `dispatchStreamEvent` to an optional `onSessionStarted` (no UI here - task
  20260724-152230 consumes it).
- Tests: mid-turn recording (revert-verified: neutering the call makes the
  mid-turn assertion fail), failed-turn recording, a bare session_meta-only
  rollout still lists as "(untitled)", and a `dispatchStreamEvent` routing unit.

Verify-first paid off: confirmed via a real rollout that `session_meta` is the
FIRST event (written at thread/start), which is what makes both the "listed
mid-turn" behavior and the failed-turn recording correct - `session_info` keeps
the row because `read_status` finds the rollout even with an empty transcript.

Difficulty: `ruff format scufris/ tests/` again reflowed unrelated files
(backends.py, test_mcp_server.py) and pre-existing lines (set_current signature,
two record_spawn_parent calls); reverted them to keep the diff focused (the flake
gate runs `ruff check`, lint-only, so the wrapped forms are fine). Also hit the
per-worktree `npm ci` step - node_modules is not shared across sprouts.

Reflection: format only the files I actually edited (`ruff format <file>...`)
rather than whole dirs, to avoid the revert dance. Otherwise smooth; the event
seam from Q1-A generalized cleanly to a control-plane (record-at-launch) use.
