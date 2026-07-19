# Review: chat UI - token-by-token text, thinking section, live events

- DATE: 20260720
- VERDICT: APPROVE (1 round)

## Scope reviewed

`web/src/common.ts` (`StreamTextDeltaEvent`/`StreamReasoningDeltaEvent`),
`web/src/agent-view.ts` (`sendChatStream` dispatch + `runStreamingTurn`),
`web/src/style.css` (`.chat__thinking`/`.chat__status`), `agent-view.test.ts`.

## Correctness

- Live-verified with a REAL app_server turn through `/api/chat/stream`: 16
  `text_delta` events assembled to a full sentence ("The sky looks blue because
  air molecules scatter blue sunlight..."). The bundle ships `onTextDelta`,
  `onReasoningDelta`, `text_delta`, `reasoning_delta`, `chat__thinking`,
  `chat__stream-body` - so the UI renders exactly what the backend streams.
- `sendChatStream` now dispatches `text_delta`/`reasoning_delta` (optional
  handlers) and, crucially, only calls `onError` on the `error` kind - UNKNOWN
  kinds are ignored, not errored. That was a latent bug: pre-change, any non-tool/
  done event hit `onError`, so switching to app_server would have mis-fired. Pinned
  by a new test that feeds reasoning+text+done and asserts the assembled text
  "Hello", the reasoning "let me think", and done.
- `runStreamingTurn` handles BOTH backends from one path: exec (no deltas -> the
  existing "working... Ns" + tool line, reply on done) and app_server (text fills
  token-by-token into a markdown body, reasoning streams into a collapsible
  `<details>` "thinking" section that appears only when reasoning arrives). The
  status label flips "working" -> "streaming" once text starts.
- Performance: the markdown re-render is throttled to one `requestAnimationFrame`
  (coalesced via a `renderQueued` flag), so a fast token stream re-renders at most
  once per frame instead of per token - the right call for markdown-per-token.
- Safety unchanged: the growing text goes through the same `renderMarkdown`
  (build-DOM, no innerHTML of model output); reasoning uses `textContent`. On done,
  the stored message uses `reply.text || streamed` so the finalized bubble matches.
- Both suites green: `npm run ci` (56 jsdom tests + build); backend untouched.

## Nits (non-blocking)

- Reasoning ("thinking") deltas were not emitted by the simple probe turns (the
  model produced no visible reasoning), so the thinking section is
  code-verified + unit-tested but not yet eyeballed on a reasoning-heavy turn.
  The plumbing is proven; a harder prompt will exercise it.
- The default backend stays `exec`, so this UI degrades gracefully (no deltas ->
  the prior timer/tool experience); token-by-token needs
  `SCUFRIS_AGENT_BACKEND=app_server`.

## Verdict

APPROVE. The chat now streams the assistant reply token-by-token, shows a live
collapsible "thinking" section for reasoning, and keeps the tool/timer feed - the
full ask, rendered from the app_server deltas and throttled for performance.
Live-verified end to end; the unknown-kind dispatch bug is fixed and pinned.
