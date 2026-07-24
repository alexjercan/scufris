# Decision: surface the codex session id at turn-start via a StreamSessionStarted event

- DATE: 20260724-152157
- STATUS: ACCEPTED
- TASK: 20260724-152157
- TAGS: decision, sessions, backend, codex, streaming

## Context

scufris records a run's session id in its registry only in the terminal
`persist` callback -> `mark_finished` (`scufris/app.py`), so a mid-turn page
refresh finds no session for a fresh codex orchestrator chat. The codex thread
id is already known early - right after `thread/start`/`thread/resume` returns
in `_stream_app_server` (`scufris/agent.py`), before the turn streams - it is
just not surfaced until the final `StreamDone`. We need the run-launch path (and,
later, a reattaching client) to learn the id at turn-start.

The backend seam between codex and the app is the `StreamEvent` async iterator
(`backend.stream(...)` yields events; `_launch_agent_turn`'s `turn_stream` already
inspects them, e.g. capturing `StreamDone.session_id`). The Q1-A change
(`AgentRunStatus.prompt` + `onUserPrompt`) just used this same seam.

## Decision

Add an additive `StreamSessionStarted(kind="session_started", session_id)` event.
`_stream_app_server` emits it the moment `new_thread_id` is set (after
thread/start|resume, before turn/start). `_launch_agent_turn`'s `turn_stream`
records ownership on it immediately via a new
`AgentStore.record_running_session(agent_id, backend, session_id)` (thin wrapper
over `registry.set`, keyed under the launch-time backend snapshot, idempotent
with the terminal `mark_finished`). The frontend routes it through
`dispatchStreamEvent` to an optional `onSessionStarted` handler (no UI change in
this task; the landing consumes it in the follow-up).

## Alternatives considered

- **A non-stream callback threaded into `backend.stream`** (e.g. an `on_session`
  kwarg). Rejected: it adds a second, backend-specific control channel alongside
  the event stream that already exists, does not ride the reattach event bus (so a
  reattaching client could not learn the id the same way), and every backend's
  `stream` signature would grow a param most ignore.
- **Keep recording only at `mark_finished`, and instead have the frontend poll a
  new "in-flight session id" field on `/status`** (like Q1-A's prompt). Rejected:
  the registry still would not own the session mid-turn, so the switcher list
  (`/api/agent/sessions`, registry-backed) still could not show it - the fix has
  to put the id in the registry, not just expose it on one endpoint.
- **Do nothing (accept the post-turn delay).** Rejected: it is the reported bug.

## Consequences

- Easier: the registry owns the session from turn-start, so `/api/agent/sessions`
  (current + list) reflects it mid-turn for free; the event also rides the reattach
  bus, so the follow-up landing task can pin the session id live with no new
  endpoint. Symmetric with `StreamDone.session_id`.
- Harder / watch: a new event kind is one more thing every stream consumer may
  need to consider (additive - unknown kinds are ignored, so existing consumers
  are safe). Recording mid-turn means a thread that starts then errors now leaves
  an owned session - acceptable and consistent with today's mark_finished-on-error
  (the thread's rollout exists on disk from thread/start), and covered by a test.
- Codex-only: other backends do not emit it (they carry the id on `done`);
  generalizing the launch-time registry timing to claude/opencode is out of scope
  for this goal (their ownership is already recorded provider-side at launch).
