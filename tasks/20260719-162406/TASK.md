# Add agent chat panel to the dashboard (streaming)

- STATUS: OPEN
- PRIORITY: 15
- TAGS: feature,backlog,agent,ui

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

- [ ] Agent continuity: extend `_run_codex_exec` to run with `--json` (+
      `--output-last-message`), parse the `thread.started` `thread_id`, and
      `codex exec resume <id>` on later turns; `CodexCliAgent` holds the thread
      id across `chat()` calls and gains a `reset()` (new conversation). Update
      the runner seam + tests. Drop `--ephemeral` so sessions persist for resume.
- [ ] Backend: `POST /api/chat` ({message} -> {reply}) driving ONE shared agent
      instance held on app state, serialized with an `asyncio.Lock`;
      `AgentUnavailable` -> HTTP 503 with the message. `POST /api/chat/reset`
      starts a new conversation. Expose `agent_enabled` in `GET /api/config`.
- [ ] Frontend: a chat panel in the dashboard SPA (message list + input + send +
      "new chat"), themed with the existing scufris CSS. POST to `/api/chat`,
      show a pending assistant bubble, render the reply; surface a disabled/error
      state from `/api/config` + 503s. Keep the transport one swappable function.
- [ ] Tests: backend `POST /api/chat` with a fake agent (reply + 503 when
      disabled) and `/api/chat/reset`; agent continuity (resume passes the thread
      id) via the fake runner; `npm run ci` green for the frontend.
- [ ] LIVE VERIFY on this host: enable the agent, send two messages in the UI (or
      via curl to `/api/chat`), confirm a real reply and that turn 2 keeps
      context. Record evidence.
- [ ] `ruff`/`mypy`/`pytest` + `npm run ci` green; update README/AGENTS if the
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
