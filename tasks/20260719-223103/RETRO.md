# Retro: agent chat - live turn progress (SSE streaming)

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Asking the user the spike's deferred a/b question (streaming vs spinner-only)
  before building was the right call - it is a real fork with a ~3x complexity
  delta and a transport change; the one-click answer aimed the whole cycle.
- The additive design kept risk low despite touching the core chat: the existing
  `_run_codex_exec` + `chat()` path is UNCHANGED (only refactored to share
  `_exec_args`/parse helpers), so `/api/chat`, the CLI, and fork kept passing
  their tests untouched. Streaming is a parallel `chat_stream`/`_stream_codex_exec`
  seam, injectable like the existing runner.
- The fake-`codex` script integration proved the whole subprocess -> SSE pipe
  without a billed model call (same pattern as the existing runner tests): tool
  frame the moment host_stats completes, then the done frame with reply/usage/
  session. Verifying the actual SSE bytes beat asserting internal state.
- Robustness handled up front: a wall-clock deadline (not per-line), a nonzero
  exit -> error frame, and a `finally` that kills the codex proc if the generator
  closes early (client disconnect) - so a dropped connection cannot orphan a codex
  process.

## What went wrong / friction

- The honest granularity limit: codex emits `item.completed` (not
  `item.started`), so the label is "ran <tool>" (post-hoc) rather than a live
  "running <tool>". Not a bug - the spinner + ticking timer carry the "still
  working" signal, and tool names accumulate as they finish. Framed it that way in
  the UI rather than pretending it is real-time.

## Lessons

- `sse-streaming-from-a-subprocess-in-fastapi` (backend/frontend): to stream a
  slow subprocess to the browser: (1) read stdout line-by-line
  (`await proc.stdout.readline()`) with a wall-clock DEADLINE, not `communicate()`;
  (2) yield events from an async generator and kill the proc in `finally` for
  early close; (3) serve via `StreamingResponse(gen(), media_type=
  "text/event-stream")` emitting `data: <json>\n\n`, holding any turn lock for the
  whole stream; (4) on the client, read `resp.body.getReader()` and parse frames
  incrementally (carry the partial-frame remainder across chunks). Keep the
  existing non-streaming path intact and additive. 20260719-223103.

## Follow-ups

- Stream the fork turn too (it currently uses the non-streaming path).
- A cancel/abort button (client `AbortController` on the fetch -> the `finally`
  kills codex) - cheap now that the plumbing exists.
- Remaining UX-review backlog: multi-line composer (223105), sidebar grouping
  (223106), chat affordances/polish (223111).
