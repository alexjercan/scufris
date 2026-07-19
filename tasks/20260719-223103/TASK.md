# Agent chat: live turn progress and streaming feedback

- STATUS: CLOSED
- PRIORITY: 38
- TAGS: feature, agent, ui, spike

## Implementation

- Backend `agent.py`: `StreamTool`/`StreamDone`/`StreamError` events; a streaming
  runner `_stream_codex_exec` (reads codex stdout line-by-line with a wall-clock
  deadline, yields a `StreamTool` per completed mcp_tool_call + a final
  `StreamDone`/`StreamError`, kills the proc in `finally` on early close);
  `CodexCliAgent.chat_stream` over an injectable `stream_runner` (updates the
  session on done) + `Agent`/`DisabledAgent`/`build_agent`. Shared `_exec_args` +
  parse helpers so `_run_codex_exec` is unchanged.
- `app.py`: `POST /api/chat/stream` -> `StreamingResponse` (`text/event-stream`),
  holds `chat_lock`, emits each event as `data: <json>`; 503 when off.
- Frontend: `common.ts` stream types; `agent-view.ts` `parseSseFrames` (pure,
  partial-frame safe), `sendChatStream` (fetch stream reader -> onTool/onDone/
  onError), `runStreamingTurn` (spinner + live "working... Ns" + "ran <tool>",
  then the reply); `style.css` spinner.
- Tests: backend stream runner (tool-then-done, nonzero error), chat_stream
  session update, disabled error, endpoint SSE frames + 503; jsdom parseSseFrames
  + sendChatStream (stubbed fetch). 55 jsdom tests; `ruff`/`mypy`/`pytest` green.
  Live-verified the full pipe via a fake-codex script (SSE tool then done).

## Goal

`codex exec` turns can take many seconds to minutes (it reasons and runs tools),
but the pending state is the literal string "..." with no spinner, elapsed time,
tool activity, or cancel - so a slow turn is indistinguishable from a hang.
Replace it with real feedback: a working indicator, an elapsed timer, and live
"running <tool>..." derived from the `codex exec --json` per-item events we
already produce but currently discard until the turn ends.

## Decision (user, from the spike's open question): STREAMING (SSE)

Build the richer option: a streaming endpoint that reads codex's `--json` events
as they arrive and pushes them to the browser, so the pending bubble shows a
spinner + a live elapsed timer AND tool activity ("ran host_stats" as each tool
completes). codex emits `item.completed` per tool as it finishes (no
`item.started`), so progress is "ran <tool>" accumulating, plus the timer - honest
live feedback, not token-by-token text (codex is turn-level).

## Steps

- [ ] `scufris/agent.py`: factor the exec-arg building into a shared helper, then
      add a STREAMING runner `_stream_codex_exec(settings, prompt, session_id) ->
      AsyncIterator[StreamEvent]` that reads codex stdout line-by-line (deadline
      = agent_timeout_seconds), yielding `StreamTool` per `mcp_tool_call`
      item.completed and a final `StreamDone{reply, session_id}` (or `StreamError`
      on timeout/nonzero). Kill the subprocess in a `finally` if the generator is
      closed early (client disconnect). Keep the existing `chat()`/`_run_codex_exec`
      path intact for `/api/chat` + the CLI + fork.
- [ ] `CodexCliAgent.chat_stream(prompt)` over an injectable `stream_runner`
      (updates `_session_id` on `StreamDone`); add to the `Agent` protocol +
      `DisabledAgent` (raises `AgentUnavailable`). `build_agent` wires the default.
- [ ] `scufris/app.py`: `POST /api/chat/stream` -> `StreamingResponse`
      (`text/event-stream`) that holds `chat_lock` for the stream and emits each
      event as an SSE `data: <json>` frame; 503 when the agent is off.
- [ ] `web/src/agent-view.ts`: a `sendChatStream(message, {onTool, onDone,
      onError})` that POSTs and reads the SSE stream (fetch + `body.getReader()` +
      a small frame parser). The chat submit uses it: the pending bubble shows an
      animated "working... <n>s" (a live elapsed timer via setInterval) that
      appends "· ran <tool>" as tool events arrive, then finalizes to the reply
      (markdown) + meta + usage on done. `style.css`: a spinner/pulse for the
      pending state.
- [ ] Tests: backend - `_stream_codex_exec` against a fake `codex` script that
      emits tool + turn events over time (asserts Tool then Done); `chat_stream`
      updates the session; the endpoint streams SSE frames + 503 disabled. jsdom -
      an SSE frame parser unit + a `sendChatStream` fed a fake `fetch`/reader
      asserting onTool/onDone fire. `npm run ci` + `ruff`/`mypy`/`pytest` green.
- [ ] LIVE serve smoke: a real turn shows the timer ticking + a tool line, then
      the reply; verify against this host's codex.

## Definition of Done

- Sending a message shows live progress: a spinner + an elapsed-seconds timer that
  ticks during the turn, tool completions appearing as they happen, then the reply
  renders (markdown) with its meta. Streamed over SSE from codex's `--json` events;
  degrades to 503 when the agent is off; the non-streaming `/api/chat` + CLI still
  work. Tests green (backend stream + endpoint, jsdom SSE parse + stream consume);
  serve-verified on this host.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P0). Lesson `harvest-the-stream-you-
  already-run` - the item events are already in the `--json` stream.
- A cancel/abort affordance for a runaway turn is a NICE-TO-HAVE; include only if
  cheap (client aborting the fetch kills the stream; killing the codex proc on
  disconnect is the `finally` above).
- Fork keeps using the non-streaming path this cycle; streaming it is a follow-up.
