# B4: per-agent chat endpoint (message -> stream, resume the agent session) + transcript

- STATUS: CLOSED
- PRIORITY: 38
- TAGS: agents,backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED


## Goal

A per-agent multi-turn CHAT endpoint: `POST /api/agents/{id}/chat` streams a turn
via `get_backend(agent.backend).stream(prompt=message, session_id=
agent.session_id, cwd=project.cwd, permission_mode=agent.permission_mode)` through
the SAME supervisor + event bus as `run`, persisting the session id (one session
per agent, resumed each turn). `GET /api/agents/{id}/transcript` returns that
session's history so the UI can rebuild the conversation.

## Steps

- [x] backends.py: add `read_transcript(settings, session_id) -> list[TranscriptMessage]`
      to the `AgentBackend` protocol + impls. Codex delegates to
      `sessions.read_transcript(resolve_codex_home, session_id)`; Claude parses
      its session JSONL via a new PURE `parse_claude_transcript(objs)` (mirrors
      `parse_claude_stream`); Mock returns `[]`. Empty when no session_id.
- [x] app.py: factor `_launch_agent_turn(agent, project, prompt) -> (run_id, EventBus)`
      out of `run_agent` (the 409-active check, session-capture closure,
      `mark_running`, `supervisor.start` on `serialize_key=agent_id`,
      `mark_finished` persist). Rewrite `run_agent` to call it (behavior
      unchanged).
- [x] app.py: `AgentChatRequest(message: str)` + `POST /api/agents/{id}/chat`:
      validate agent (404) + non-empty message (422) + project (422), launch the
      turn (409 if a run/chat is already active), and relay the run's bus INLINE
      as SSE (same frame shape as `/api/chat/stream` and `/events`), so the
      caller streams its own turn. Resumes `agent.session_id`; the persist
      callback writes the (possibly new) session id back.
- [x] app.py: `GET /api/agents/{id}/transcript` -> `TranscriptResponse` via
      `get_backend(agent.backend).read_transcript(settings, agent.session_id)`
      (empty list when the agent has never run).
- [x] Tests (mock + fixtures): chat streams a turn and persists the session id
      (reaches done, `/status` shows it); a second chat while one is active ->
      409; empty message -> 422; unknown agent -> 404. Transcript: a claude
      session JSONL fixture round-trips through the endpoint; mock agent -> empty.
      Plus a pure `parse_claude_transcript` unit test on captured-shape JSONL.

## Definition of Done

- `POST /api/agents/{id}/chat` streams a turn as SSE and persists the resumed
  session id (test: `test_agent_chat_streams_and_persists_session`).
- The chat turn goes through the same supervisor/bus as run: `/status` and
  `/events` reflect it, and a concurrent turn is refused
  (test: `test_agent_chat_conflicts_with_active_run` -> 409;
  `test_agent_chat_validates` -> 422/404).
- `GET /api/agents/{id}/transcript` returns the session's history
  (test: `test_agent_transcript_reads_claude_session`;
  `test_agent_transcript_empty_for_unrun_agent`).
- `parse_claude_transcript` maps a claude session JSONL to TranscriptMessages
  (test: `test_parse_claude_transcript`).
- Full check suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation B4). The backends already resume by session_id.
- Depends on: 20260721-112430 (B2), 20260721-112432 (B3).
- Reuse map (from an out-of-context exploration): `run_agent` (app.py:834-904)
  is the template - chat is the same machinery with a MESSAGE instead of a goal
  and no goal-required 422. Supervisor.start returns the EventBus; `agent_runs`
  keyed by agent id already feeds `/status` + `/events`, so a chat turn reuses
  both for free. Backend `stream(...)` yields StreamDone(session_id=...) which
  the persist closure writes via `mark_finished`. `TranscriptMessage`/
  `TranscriptResponse` already exist (sessions.py:142, app.py:377); codex uses
  `read_transcript`, claude reads `<claude_home>/projects/**/<sid>.jsonl` via
  `_find_claude_session` + `_iter_jsonl`.
- F4 (next) reuses the agent-view chat helpers against these endpoints.
- Close-out: chat reused the run machinery near-verbatim - the extracted
  `_launch_agent_turn` is the whole conflict/session/persist story, so `run` and
  `chat` differ only in goal-vs-message and the goal-required 422. The chat
  endpoint relays the bus INLINE (like `/api/chat/stream`) rather than returning
  RunStarted, since the caller streams its own turn; `/events` + `/status` still
  work for reconnection because it shares `agent_runs`. Added a per-backend
  `read_transcript` (codex -> `sessions.read_transcript`; claude -> pure
  `parse_claude_transcript` over the session JSONL; mock -> []) so the transcript
  endpoint is backend-agnostic. e2e (real uvicorn): `/chat` streams
  text_delta + done(session_id), empty msg -> 422, `/transcript` empty pre-turn,
  and neither `/{id}` nor `/backends` is shadowed.
- Difficulty: the first 409 test used sync `TestClient.stream` + a second
  request to catch the active run, but BOTH TestClient and httpx ASGITransport
  BUFFER the whole SSE body before returning, so holding a turn open that way
  DEADLOCKED pytest (hung > 3 min, killed). Rewrote it as an async test firing
  concurrent requests on one loop via `httpx.AsyncClient(ASGITransport)`: a
  `create_task` first turn blocked on an `asyncio.Event`, poll `/status` to
  "running", then a second POST returns 409, then release. Ledgered.
