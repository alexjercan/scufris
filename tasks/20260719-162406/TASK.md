# Add agent chat panel to the dashboard (streaming)

- PRIORITY: 15
- TAGS: feature, backlog, agent, ui
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Add the agent chat panel to the dashboard: chat with the Scufris agent from the
UI, with streaming replies.

## Reality check (from probing codex exec, 20260719)

- `codex exec --json` emits turn-level events (`thread.started` with a
  `thread_id` UUID, `turn.started`, `item.completed`, `turn.completed`), NOT
  token deltas. So codex-exec replies are **turn-based, not token-streamed**.
  "Streaming" here = send the message, show a pending state, render the full
  reply when the turn completes. Token streaming is not available via this tool;
  noted honestly, not faked.
- Multi-turn continuity DOES work: capture `thread_id` from turn 1's `--json`
  output, then `codex exec resume <thread_id> "<msg>"` for later turns.

## Steps

- [x] Agent continuity: extend `_run_codex_exec` to run with `--json` (+
      `--output-last-message`), parse the `thread.started` `thread_id`, and
      `codex exec resume <id>` on later turns; `CodexCliAgent` holds the thread
      id across `chat()` calls and gains a `reset()` (new conversation). Update
      the runner seam + tests. Drop `--ephemeral` so sessions persist for resume.
- [x] Backend: `POST /api/chat` ({message} -> {reply}) driving ONE shared agent
      instance held on app state, serialized with an `asyncio.Lock`;
      `AgentUnavailable` -> HTTP 503 with the message. `POST /api/chat/reset`
      starts a new conversation. Expose `agent_enabled` in `GET /api/config`.
- [x] Frontend: a chat panel in the dashboard SPA (message list + input + send +
      "new chat"), themed with the existing scufris CSS. POST to `/api/chat`,
      show a pending assistant bubble, render the reply; surface a disabled/error
      state from `/api/config` + 503s. Keep the transport one swappable function.
- [x] Tests: backend `POST /api/chat` with a fake agent (reply + 503 when
      disabled) and `/api/chat/reset`; agent continuity (resume passes the thread
      id) via the fake runner; `npm run ci` green for the frontend.
- [x] LIVE VERIFY on this host: enable the agent, send two messages in the UI (or
      via curl to `/api/chat`), confirm a real reply and that turn 2 keeps
      context. Record evidence.
- [x] `ruff`/`mypy`/`pytest` + `npm run ci` green; update README/AGENTS if the
      surface changed.

## Definition of Done

- The running dashboard has a chat panel; sending a message returns the agent's
  real reply, and a follow-up keeps conversation context (verified live on this
  host). With the agent off, the panel shows a clear disabled state.
- `POST /api/chat` + `/api/chat/reset` exist and are serialized; tests green with
  the agent faked; frontend `npm run ci` green.

## Notes

- Spike: tasks/20260719-153040/SPIKE.md.
- Frontend: a chat panel in the existing web/ single-page app (web/src), styled
  with the existing scufris theme; sends messages to a backend chat endpoint and
  renders streaming assistant replies (SSE or chunked). Keep the transport a
  single swappable seam like the stats polling.
- Backend: a FastAPI chat endpoint that drives the Agent interface from
  tatr 20260719-162356 (turns/threads); stream tokens back to the client.
- Show tool activity when the agent runs a tool (e.g. a tatr command) so the
  user can see what it did (ties to the MCP tool server, tatr 20260719-162419).
- Depends on the agent backend (tatr 20260719-162356). Pairs with the tool
  server for visible tool calls.

## Implementation

- Agent continuity: `_run_codex_exec` now runs with `--json` (+
  `--output-last-message`), parses the `thread.started` `thread_id`, and
  `codex exec resume <id>` on later turns. Returns a `TurnOutcome(text,
  thread_id)`; `CodexCliAgent` remembers the thread across `chat()` and gains
  `reset()`. Dropped `--ephemeral` so sessions persist. Gotcha found live:
  `codex exec resume` inherits the original session's sandbox and REJECTS
  `--sandbox`, so that flag is only passed on the first turn.
- Backend (`scufris/app.py`): `POST /api/chat` ({message} -> AgentReply) driving
  one shared agent held in `create_app`, serialized with an `asyncio.Lock`;
  `AgentUnavailable` -> HTTP 503. `POST /api/chat/reset` starts a new
  conversation. `GET /api/config` now returns `AppConfig` incl. `agent_enabled`.
- Frontend (`web/src`): a themed chat panel (message list + input + send + new
  chat) under the stat cards; POSTs to `/api/chat`, shows a pending bubble, and
  renders the reply; disabled state when `agent_enabled` is false; error bubble
  on failure. One `sendChat` transport function. Chat CSS added to `style.css`.
- Tests: backend `POST /api/chat` (reply + 503 when disabled) and
  `/api/chat/reset`; agent continuity (resume passes the captured thread id, and
  `reset` clears it) via the fake runner; the fake-codex integration test now
  emits a `thread.started` event and asserts the parsed id. ruff+mypy+pytest and
  `npm run ci` green.

### Live verification (DoD)

On this host (`SCUFRIS_AGENT_ENABLED=1`), via `POST /api/chat`: turn 1 "remember
codeword BANANA" -> `ok`; turn 2 "what is the codeword?" -> `BANANA` (context
kept across turns); after `/api/chat/reset`, the same question -> `unknown`
(fresh conversation). Real GPT-5.5, end to end. `GET /` serves the panel;
`/api/config` reports `agent_enabled: true`. Streaming note: `codex exec` events
are turn-level, so replies are turn-based (pending state -> full reply), not
token-streamed - documented, not faked.
