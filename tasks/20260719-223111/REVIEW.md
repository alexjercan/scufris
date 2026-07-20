# Review: chat message affordances and polish

- VERDICT: APPROVE
- ROUND: 1

## Summary

All five spike items landed in one cohesive cycle: whole-reply copy button,
per-message timestamps (live + historical, the latter plumbed through a new
`TranscriptMessage.ts` from each rollout event's timestamp), no-yank scroll with
a "new messages" pill, an onboarding empty state with clickable example prompts,
and `aria-live` + focus a11y. 72 frontend tests + 123 pytest green; the built
`dist/index.html` ships the aria-live log and the pill.

## What is good

- The scroll model is a proper follow-state (`_stickToBottom` maintained by a log
  scroll listener) with a `_rendering` guard so the rebuild's own scroll events
  do not mis-read the follow state off a transiently-short log. A user action
  (send/fork/switch) re-pins. This is the correct shape, not a hack.
- Timestamps are real on both paths: live turns stamp `Date.now()`, and switching
  to an old session shows true times because the backend now reads each event's
  top-level `timestamp`. Verified the rollout format actually carries it before
  plumbing (not assumed).
- Copy reuses the same clipboard-guarded pattern as the code-block copy; `getText`
  is read lazily so it always copies the current raw markdown.
- Good jsdom discipline: the layout-dependent bits (`isNearBottom`, the pill
  reveal, scroll preservation) are unit-tested by defining scroll metrics and
  dispatching real scroll events, and the pure helpers (`formatTimestamp`) are
  tested directly.

## Findings

- FIXED in-review (was a real regression I introduced) - `renderLog`'s
  `replaceChildren()` resets `scrollTop` to 0, and the not-following branch did
  not restore it, so a scrolled-up reader receiving a new reply was yanked to the
  TOP of the history (worse than the original yank-to-bottom). Fixed by capturing
  `prevTop` before the rebuild and restoring it in `maybeScroll` when not
  following; added a regression test (asserts scrollTop stays put, not 0 and not
  scrollHeight).
- KNOWN LIMITATION (noted, not blocking) - `aria-live="polite"` sits on the whole
  `#chat-log`, which `renderLog` rebuilds via `replaceChildren`. With
  `aria-relevant="additions"` a screen reader may re-announce the whole
  conversation on a turn, not just the new reply. The task asked for an aria-live
  log and that is delivered; a surgical "announce only the latest reply" region is
  a worthwhile follow-up but a larger rendering change. Recorded in the retro.
- MINOR - copy/edit reveal on `.chat__foot:hover` (or focus), not on hovering the
  message bubble itself. The always-present timestamp gives a hover target, but
  bubble-hover reveal would be more discoverable. Deferred (the assistant/meta/foot
  adjacency makes the CSS selector fiddly); focus-reveal keeps it keyboard-
  accessible regardless.

## Verdict

APPROVE. The scroll regression - the one real bug - was caught and fixed in
review with a pinning test. The aria-live over-announcement is an honest,
documented limitation of the whole-log rebuild model, not a defect in this
change, and is filed as a follow-up.
