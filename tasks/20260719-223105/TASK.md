# Agent chat: multi-line composer with Enter/Shift-Enter

- PRIORITY: 30
- TAGS: feature, agent, ui, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

The prompt composer is a single-line `<input>`. You instruct an agent with detail
(multi-paragraph asks, pasted logs), so replace it with an autosizing `<textarea>`
that grows with content (to a max height then scrolls), with Enter = send and
Shift+Enter = newline, and a clear sending/disabled state. Keep the send button.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P1).
- Preserve the existing submit path (`sendChat`) and the disabled-while-sending
  behavior; just change the control + key handling. Keep it side-effect-free for
  jsdom where practical.

## Implementation

- `index.html`: `<input type="text">` -> `<textarea rows="1">` (same id/class),
  placeholder now spells out "Enter to send, Shift+Enter for newline".
- `agent-view.ts`: `initChat`/`runStreamingTurn` retype the control to
  `HTMLTextAreaElement`. Extracted a shared `submit()` used by both the form
  submit and a new `keydown` handler: Enter (no shift, not IME-composing) ->
  `preventDefault` + submit; Shift+Enter falls through to the textarea's native
  newline. `submit()` guards on `input.disabled` so Enter is a no-op mid-turn.
  New `autosizeComposer()` grows the textarea to fit content up to
  `COMPOSER_MAX_HEIGHT` (200px) then scrolls; wired to `input` events, the send
  path, and `stop()` (so it shrinks back to one row when cleared/re-enabled).
- `style.css`: `.chat__input` gets `resize: none`, `line-height`, `max-height`,
  `overflow-y: hidden`; `.chat__form` aligns items to `flex-end` so the send
  button stays bottom-pinned as the composer grows; `.chat__send` gets a
  `min-height` to match a single row.

## Tests

- `agent-view.test.ts`: existing incremental-render markup switched to
  `<textarea>`. New "multi-line composer (initChat)" block: Enter sends + clears
  + disables + posts the user message; Shift+Enter does NOT send (value kept,
  default allowed); Enter is ignored while a turn is in flight; whitespace-only
  input does not send. 62 frontend tests green, webpack build clean.
- Autosize height depends on layout (`scrollHeight`), which jsdom does not
  compute, so it is verified by eye in the served bundle, not asserted in jsdom
  (per `frontend-verify-needs-e2e-serve`).
