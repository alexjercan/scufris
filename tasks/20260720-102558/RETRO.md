# Retro: message affordances visible at rest

- DATE: 20260720
- VERDICT: shipped

## What went well

- The fix was exactly as small as the diagnosis promised: a one-line resting
  opacity change (0 -> 0.6) plus adding `:hover` to the brighten selector. No
  markup or JS, layout untouched, so essentially zero regression surface.
- Grepping the built `dist/agent.js` for the shipped `opacity: 0.6` / `1` rules
  gave a real artifact check for something jsdom cannot compute - a cheap way to
  prove the CSS actually shipped rather than trusting the source edit.

## What went wrong / friction

- Nothing of substance. The only judgment call was the "pin it with a test" ask on
  a CSS-only change: jsdom has no cascade for the webpack stylesheet, so the honest
  pin is the structural invariant (buttons rendered, not `hidden`, no hover needed)
  plus an eyeball/grep check of the opacity - not a computed-style assertion.

## Lesson

- No new ledger entry: this reused `frontend-verify-needs-e2e-serve` (CSS visual
  state is eyeball/artifact-verified, not jsdom-asserted). Worth restating that a
  "test the visual state" request on a pure-CSS change means testing the DOM-level
  invariant + grepping the built bundle, not faking a computed-style check jsdom
  cannot honor.

## Follow-ups

- None. The sibling round-2 tasks (102559 tool-steering, 102600 head, 102601
  settings, 102602 polish) remain open.
