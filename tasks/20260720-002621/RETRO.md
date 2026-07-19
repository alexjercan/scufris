# Retro: chat UI - token-by-token, thinking, events

- DATE: 20260720
- VERDICT: APPROVE (1 review round)

## What went well

- The backend task did the hard part (proving + emitting the deltas), so this was
  a clean consume: add the two event types, dispatch them, and render. The live
  app_server turn (16 token deltas -> a full sentence in the bubble) confirmed the
  whole pipe end to end.
- One code path serves both backends: exec (no deltas) keeps the timer/tool UI and
  fills the reply on done; app_server streams tokens into the same bubble +
  reasoning into a `<details>`. No branching on backend in the UI - it just reacts
  to whatever events arrive, which is the robust shape.
- Throttling the markdown re-render to one `requestAnimationFrame` (coalesced) was
  the key perf move: re-rendering markdown per token would thrash the DOM; per
  frame is smooth and cheap.

## What went wrong / friction

- Found a latent bug while wiring: `sendChatStream`'s dispatch did
  `else handlers.onError(event.detail)` for anything not tool/done - so once
  app_server started emitting `text_delta`, every token would have called
  `onError(undefined)`. Fixed to error ONLY on the `error` kind and ignore unknown
  kinds. Pinned by the new delta-dispatch test. A good reminder that an
  exhaustive-looking if/else over a union's `kind` silently mis-handles new
  variants.

## Lessons

- `dispatch-only-known-kinds-not-else-error` (frontend): when switching on a
  discriminated union's `kind`, do NOT put the error/fallback in the final `else`
  - a new variant then routes to the error path. Match each known kind explicitly
  (incl. `error`) and IGNORE the unknown, so adding a variant is additive.
  20260720-002621.

## Follow-ups

- Eyeball the "thinking" section on a reasoning-heavy prompt (the simple probe
  turns emitted no reasoning deltas; the plumbing is unit-tested + code-verified).
- Verify the tool-event chips on a real tool-calling app_server turn (carried from
  the backend task).
- Optional: a small "backend: app_server" indicator, and a per-model context bar
  now that thread/tokenUsage is available.
