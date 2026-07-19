# Review: agent chat - live turn progress and streaming feedback (SSE)

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

Backend: `scufris/agent.py` (`StreamTool`/`StreamDone`/`StreamError`,
`_stream_codex_exec`, `chat_stream`, shared `_exec_args`/`_parse_event_line`/
`_usage_from`/`_tool_call_from`), `scufris/app.py` (`POST /api/chat/stream`).
Frontend: `common.ts` stream types, `agent-view.ts` (`parseSseFrames`,
`sendChatStream`, `runStreamingTurn` + submit swap), `style.css` (spinner). Tests
on both sides.

## Correctness

- End-to-end live-verified with a fake `codex` script (full subprocess -> SSE
  pipe, no billed model call): `POST /api/chat/stream` returned `text/event-stream`
  and emitted a `tool` frame the moment `host_stats` completed, then a `done`
  frame with the reply text, tool_calls, usage and `session_id`. Exactly the
  design.
- The streaming runner reads codex stdout line-by-line with a wall-clock DEADLINE
  (not a per-line reset), so a stalled turn becomes a `StreamError` timeout; a
  nonzero exit becomes `StreamError` with stderr; and a `finally` kills the
  subprocess if the generator is closed early (client disconnect) - no orphaned
  codex process. Verified by tests (tool-then-done, nonzero-exit error).
- Zero behavior change to the existing paths: `_run_codex_exec` was refactored to
  share `_exec_args` but its output is unchanged, so `/api/chat`, the CLI, and
  fork still work (their tests pass untouched). Streaming is additive.
- The endpoint holds `chat_lock` for the whole stream (turns stay serialized) and
  503s before opening the stream when the agent is off (tested). `AgentUnavailable`
  raised mid-stream (e.g. missing codex bin) is caught and sent as an error frame.
- Frontend: `parseSseFrames` is pure and handles partial frames across chunks
  (carries the remainder) and ignores malformed data (tested). `sendChatStream`
  reads the fetch `ReadableStream` and dispatches onTool/onDone/onError (tested
  with a stubbed fetch + real ReadableStream). The pending bubble shows a CSS
  spinner + a live "working... Ns" that ticks (setInterval) and appends "· ran
  <tool>" as tools complete; on done it finalizes to the markdown reply + meta and
  refreshes the sidebar; the interval is always cleared in `stop`/`fail`.
- Both suites green: `npm run ci` (55 jsdom tests + build), `ruff`/`ruff format`/
  `mypy`/`pytest`. Bundle ships `sendChatStream`/`parseSseFrames`/`chat__spinner`.

## Nits (non-blocking)

- codex emits `item.completed` (not `item.started`), so the label reads "ran
  <tool>" (post-completion) rather than "running" - honest given the granularity;
  the spinner + timer carry the "still working" signal.
- Fork still uses the non-streaming path (documented follow-up); the main chat
  submit is what streams.
- A real (billed) codex turn was not run here; the fake-codex integration
  exercises the whole pipe, and the live look is the user's eyeball.

## Verdict

APPROVE. The dead "..." is gone: a message now shows a spinner + a live elapsed
timer and tool completions as they happen, streamed over SSE from codex's own
`--json` events, then the reply renders. Robust (deadline, disconnect-kill,
serialized), additive to the existing chat, and tested end to end.
