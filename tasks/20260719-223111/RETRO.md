# Retro: chat message affordances and polish

- DATE: 20260720
- VERDICT: shipped

## What went well

- Doing all five items in one cycle was the right call: they are all footer/log
  polish on the same render path, so batching kept the diff coherent (one `LogEntry`
  shape, one `renderLog`, one CSS block) instead of five tiny cycles re-touching
  the same functions.
- Verified the rollout event format actually carries a top-level `timestamp`
  BEFORE plumbing `TranscriptMessage.ts` (grepped a real `~/.codex` rollout), so
  historical timestamps are real, not a hopeful field that ends up always null.
- The self-review earned its keep: it caught a scroll regression the tests could
  not (jsdom has no layout), by reasoning about what `replaceChildren` does to
  `scrollTop` in a real browser.

## What went wrong / friction

- The no-yank scroll had a subtle self-inflicted bug: I gated the auto-scroll on a
  follow-state, but `renderLog` rebuilds the whole log with `replaceChildren`,
  which resets `scrollTop` to 0. The not-following branch did nothing, so a
  scrolled-up reader was flung to the TOP on every new reply - a worse yank than
  the one I set out to remove. jsdom could not reproduce it (scrollTop is a static
  0 there), so only reasoning caught it. Fixed by capturing `prevTop` before the
  rebuild and restoring it.
- `aria-live` on the whole log is a blunt instrument: because `renderLog` replaces
  all children, a screen reader can re-announce the entire conversation per turn.
  Delivered the task's literal ask but filed the surgical version as a follow-up.

## Lessons

- `full-rebuild-render-resets-scrolltop` - any render that does
  `container.replaceChildren()` throws away the scroll position (scrollTop -> 0).
  A "don't yank the user" scroll policy must therefore CAPTURE scrollTop before the
  rebuild and RESTORE it when not auto-scrolling - skipping the scroll is not
  enough, because the rebuild itself already moved them (to the top). jsdom cannot
  catch this (scrollTop is a static 0), so reason about it or test in a browser.
- `aria-live-on-a-rebuilt-region-over-announces` - putting `aria-live` on a
  container that is re-rendered via `replaceChildren` makes assistive tech treat
  the whole thing as new each time. For "announce the new reply", the live region
  should wrap only the incrementally-appended content, not a wholesale-replaced
  log. Delivered the coarse version; the surgical one is the real fix.

## Follow-ups

- Surgical aria-live: announce only the latest assistant reply (a dedicated live
  region or an incremental append that does not replaceChildren the whole log).
- Reveal copy/edit on message-bubble hover too, not just footer hover (the
  assistant/meta/foot adjacency makes the pure-CSS selector fiddly; may want a
  small JS class toggle).

## Family note (spike 20260719-223054)

This closes the LAST of the five tasks the agent-UX spike seeded. The whole
"fix the conversation loop" arc (markdown, streaming, composer, sidebar,
affordances) has now landed.
