# Agent backend via codex app-server: stream token/reasoning/tool deltas

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature, agent, spike

## Goal

`codex exec` cannot stream token-by-token (proven in the spike). Add a second
agent backend that drives the experimental `codex app-server` (a persistent
JSON-RPC-over-stdio server) and consumes its notification stream, so a turn yields
live events: `outputDelta` (assistant text token-by-token),
`ReasoningTextDelta`/`ReasoningSummaryTextDelta` ("thinking"), thread-item / tool /
process events, and final usage. Wire it behind the existing `Agent` seam,
config-gated (`agent_backend = exec | app_server`) so `codex exec` stays the
fallback + CLI path. Forward the new events over the existing `/api/chat/stream`
SSE endpoint as new event kinds (text-delta, reasoning-delta, event, done).

## Probe result (DONE - protocol confirmed live)

Drove `codex app-server` (stdio, newline-delimited JSON-RPC) from Python end to
end: `initialize` (clientInfo, capabilities:null) -> `thread/start`
(`{sandbox:"read-only"}` -> `{thread:{id}}`) -> `turn/start`
(`{threadId, input:[{type:"text",text,text_elements:[]}]}`). The turn/start
RESPONSE returns immediately; the stream arrives as NOTIFICATIONS:
`item/agentMessage/delta {threadId,turnId,itemId,delta}` (token-by-token text -
23 events assembled to "1, 2, 3, ... 8."), `item/reasoning/textDelta` (thinking),
`item/started`/`item/completed`, `item/mcpToolCall/*` (tools), `turn/completed`,
`thread/tokenUsage/updated`. `thread/resume {...}` continues an existing thread
(multi-turn). chatgpt auth + read-only sandbox both work. Probe:
`$CLAUDE_JOB_DIR/tmp/appserver_probe.py`.

## Steps

- [ ] `config.py`: `agent_backend: Literal["exec","app_server"] = "exec"`.
- [ ] `agent.py`: new event kinds `StreamTextDelta{delta}` and
      `StreamReasoningDelta{delta}` in the `StreamEvent` union; a pure
      `_appserver_event(obj) -> StreamEvent | None` mapping a notification to an
      event (text/reasoning delta, tool, done, error).
- [ ] `agent.py`: `_stream_app_server(settings, prompt, thread_id) ->
      AsyncIterator[StreamEvent]` - async subprocess drives the JSON-RPC
      (`codex app-server` + `_mcp_overrides` for the Scufris tools):
      initialize -> `thread/resume` if `thread_id` else `thread/start` ->
      `turn/start`; read notifications line-by-line (wall-clock deadline), yield
      text/reasoning deltas + tool events, assemble the full text, and finish with
      `StreamDone{reply(text,tool_calls,usage), session_id=thread_id}` on
      `turn/completed` (or `StreamError`). Kill the proc in `finally`.
- [ ] `build_agent`: pick `stream_runner = _stream_app_server` when
      `agent_backend=="app_server"`, else `_stream_codex_exec`. `chat()` (CLI /
      fork / `/api/chat`) stays on `_run_codex_exec`. The SSE endpoint forwards the
      new event kinds unchanged (`model_dump_json`).
- [ ] Tests: `_appserver_event` mapping (each notification -> the right event); an
      integration test with a fake `codex` script that speaks the JSON-RPC
      handshake + emits delta notifications, asserting `_stream_app_server` yields
      text deltas then done; `build_agent` backend selection. `ruff`/`mypy`/`pytest`
      + a live smoke of a real app_server turn (token deltas over SSE).

## Definition of Done

- With `SCUFRIS_AGENT_BACKEND=app_server`, a chat turn streams `text-delta` events
  (token-by-token), `reasoning-delta` (thinking), and tool events, ending in
  `done` with the assembled reply + usage + thread id (multi-turn via
  `thread/resume`); the exec backend + CLI + fork still work under the default.
  Experimental protocol pinned behind the flag. Tests green; live-verified.

## Notes

- Spike: tasks/20260720-002611/SPIKE.md.
- PLAN PROBE-FIRST: the opening step is a throwaway-grade prototype that completes
  the app-server JSON-RPC handshake (initialize + start a turn) and captures a
  real streamed turn's `outputDelta`/reasoning deltas, to confirm the protocol +
  chatgpt auth + sandbox wiring BEFORE building the production client. Use
  `codex app-server generate-json-schema` / `generate-ts` for the method + event
  shapes.
- EXPERIMENTAL protocol: pin behind the config flag, keep `codex exec` as
  fallback, and re-inspect the schema on codex upgrades.
- Depends on nothing hard; blocks tatr 20260720-002621 (the UI).
- Keep the injectable-runner/seam patterns and the SSE streaming shape from tatr
  20260719-223103; add event kinds rather than replacing them.
