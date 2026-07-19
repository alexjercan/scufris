# Chat UI: token-by-token text, thinking section, live event feed

- STATUS: OPEN
- PRIORITY: 35
- TAGS: feature, agent, ui, spike

## Goal

Render the new streaming events from the app-server backend so the user sees a
turn unfold live: the assistant bubble fills in TOKEN BY TOKEN from `text-delta`
events (markdown re-rendered as it grows); a collapsible/live "thinking" section
that streams reasoning deltas; and a live event feed of tool calls / plan updates
/ process output as they arrive. On done, finalize to the stored message + meta.

## Steps

- [ ] `common.ts`: add `StreamTextDeltaEvent`/`StreamReasoningDeltaEvent`
      (`kind: "text_delta"|"reasoning_delta"`, `delta`) to the `StreamEvent` union.
- [ ] `agent-view.ts`: `sendChatStream` dispatches the new kinds
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
- [ ] `agent-view.test.ts` (jsdom): `sendChatStream` fed a stubbed reader with
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
