# Chat UI: token-by-token text, thinking section, live event feed

- STATUS: CLOSED
- PRIORITY: 35
- TAGS: feature, agent, ui, spike

## Implementation

- `common.ts`: `StreamTextDeltaEvent`/`StreamReasoningDeltaEvent` in the union.
- `agent-view.ts`: `sendChatStream` dispatches `text_delta`/`reasoning_delta`
  (optional handlers) and now only errors on the `error` kind (unknown kinds
  ignored - fixes a latent mis-fire). `runStreamingTurn` fills the pending bubble
  token-by-token (markdown re-render throttled to one rAF), streams reasoning into
  a collapsible `<details>` "thinking" section, keeps the tool/timer feed, and
  works for both exec (no deltas) and app_server. `style.css`: thinking/status/
  stream-body.
- Tests: `sendChatStream` dispatches text+reasoning deltas -> "Hello"/"let me
  think"/done (56 jsdom total). Live-verified a REAL app_server turn: 16 token
  deltas -> a full sentence. `npm run ci` green.

## Goal

Render the new streaming events from the app-server backend so the user sees a
turn unfold live: the assistant bubble fills in TOKEN BY TOKEN from `text-delta`
events (markdown re-rendered as it grows); a collapsible/live "thinking" section
that streams reasoning deltas; and a live event feed of tool calls / plan updates
/ process output as they arrive. On done, finalize to the stored message + meta.

## Steps

- [x] `common.ts`: add `StreamTextDeltaEvent`/`StreamReasoningDeltaEvent`
      (`kind: "text_delta"|"reasoning_delta"`, `delta`) to the `StreamEvent` union.
- [x] `agent-view.ts`: `sendChatStream` dispatches the new kinds
      (`onTextDelta`/`onReasoningDelta`); unknown kinds are ignored (not errors).
- [ ] `agent-view.ts` `runStreamingTurn`: the pending assistant bubble fills in
      token-by-token from text deltas - accumulate the text and re-render its
      markdown THROTTLED (requestAnimationFrame/coalesced) for perf; a collapsible
      "thinking" section streams reasoning deltas; tool events append live chips.
      On done, finalize to the stored message (+ meta) and stop. Works for BOTH
      backends: exec (no deltas -> the current timer/tool UI, reply on done) and
      app_server (live tokens + thinking).
- [ ] `style.css`: `.chat__thinking` (distinct, muted, collapsible), a token
      cursor/typing affordance.
- [x] `agent-view.test.ts` (jsdom): `sendChatStream` fed a stubbed reader with
      `text_delta`+`reasoning_delta`+`done` frames dispatches each handler; the
      streamed text assembles. `npm run ci` green + a live app_server smoke.

## Notes

- Spike: tasks/20260720-002611/SPIKE.md.
- Depends on tatr 20260720-002619 (the app-server backend + SSE event kinds - CLOSED).
- Build on the existing SSE consumer (`sendChatStream`/`parseSseFrames`) and the
  markdown renderer (re-render the growing text; consider debouncing re-render for
  performance on fast token streams). Keep the reasoning section visually distinct
  from the answer and collapsible (it can be long).
- Escape everything; render side-effect-free where practical for jsdom; keep the
  non-app-server (exec) path working with its existing tool-chip + timer UI.
