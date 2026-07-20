# Review: multi-line composer with Enter/Shift-Enter

- VERDICT: APPROVE
- ROUND: 1

## Summary

`<input type="text">` -> autosizing `<textarea>`. Enter sends, Shift+Enter
newlines, the composer grows to a 200px cap then scrolls, and the send/disabled
behavior is preserved. Diff is small and localized to the composer; the
edit-to-fork inline editor (its own `chat__editor-input` textarea) is untouched,
and `chat-input` is referenced only in `initChat`. 62 frontend tests green,
webpack build clean, textarea confirmed in the built `dist/index.html`.

## What is good

- Shared `submit()` extracted so the form-submit and the Enter keydown drive the
  exact same path - no duplicated send logic to drift.
- `submit()` guards on `input.disabled`, so Enter (and a send-button click) are
  genuine no-ops mid-turn - correctness does not depend on the value already
  being cleared.
- `isComposing` guard on Enter: an IME candidate committed with Enter will not
  fire a half-typed message. Good detail for a non-obvious failure mode.
- `autosizeComposer` resets to `auto` before measuring, so the box SHRINKS on
  delete/clear, not just grows; wired to input, the send path, and `stop()`.
- Tests cover the real behavior branches: Enter sends+clears+disables+posts,
  Shift+Enter keeps the value and allows the default, Enter ignored while busy,
  whitespace-only does not send. The `dispatchEvent` return value is asserted to
  prove `preventDefault()` fired.

## Findings (all minor, non-blocking)

- MINOR - the 200px cap lives in two places: `COMPOSER_MAX_HEIGHT` (JS) and
  `.chat__input { max-height: 200px }` (CSS). They must stay in sync. Acceptable
  (the CSS is a backstop and the JS comment names the constant), but a future
  change to one must touch the other.
- MINOR - the send button is not visually disabled during a turn (only the
  textarea is, as before). Correctness is covered by the `submit()` guard; a
  disabled/greyed send button would be a nicer signal. Out of scope for this
  task; belongs with the affordances/polish task (20260719-223111).
- MINOR (test hygiene) - the composer tests use an open stream that never
  completes, so `runStreamingTurn`'s 500ms `paintStatus` interval is left
  running (its `stop()` never fires). Harmless (fires on a detached node, cheap)
  and consistent with the file's existing streaming test, but not tidy.

## Verdict

APPROVE. The change is correct, well-tested at the logic layer, and the autosize
(which jsdom cannot measure) is appropriately left to eyeball verification per
`frontend-verify-needs-e2e-serve`. The three findings are polish, not defects.
