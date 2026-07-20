# Retro: grouped, labeled sidebar sections

- DATE: 20260720
- VERDICT: shipped

## What went well

- The spike's diagnosis ("the head `ctx · out` is redundant with the context
  box") turned out to be verifiable from the backend, not just a vibe: reading
  `sessions.py:read_context` showed the box's `output` is the cumulative session
  total from disk and its fill is last-turn input - strictly better than the head
  counter, which only summed turns done in the current browser tab. That turned a
  "should we dedupe?" judgment call into a clear delete-the-worse-one.
- Deleting the indicator removed a whole cluster of state (`_cumulativeOutput`,
  `_lastContext`, `applyUsage`, `resetUsage`) instead of hiding a widget. The
  render layer got smaller. It was safe because every flow already funnels
  through `refreshSidebar()`, the single authoritative render.
- Reusing `.usage-block__head`'s styling for the new `.sidebar__label` (one
  grouped selector) kept the three boxes visually one family with almost no new
  CSS.

## What went wrong / friction

- Naming the window descriptor `const window` in `renderUsage` shadowed the
  global `window` (used for `window.confirm` / `window.setTimeout` elsewhere in
  the file). No runtime bug and eslint did not flag it, but it is a real footgun;
  caught in self-review and renamed to `windowLabel`. A domain word ("window"
  from `window_minutes`) collided with a browser global.

## Lessons

- `dont-shadow-browser-globals-with-domain-words` - a local named `window`,
  `document`, `name`, `status`, `length` etc. shadows a global that other code in
  the same module relies on. eslint's default config does NOT catch it. When a
  domain concept (here a rate-limit "window") wants that name, suffix it
  (`windowLabel`). Cheap to avoid, annoying to debug.
- `prefer-one-authoritative-render-over-a-parallel-client-counter` - the head
  indicator was a client-side accumulator shadowing data the API already returns
  authoritatively. Two sources of the same number drift (the counter only saw
  this tab's turns). When an endpoint already carries the truth and every mutation
  path refreshes from it, delete the parallel counter rather than syncing it.

## Follow-ups

- This removal RETIRES the premise of `separate-usage-reset-from-log-reset`
  (20260719-224101): `resetUsage` no longer exists, so the "keep a narrow
  resetUsage distinct from the full reset" guidance has no referent. Annotated in
  the ledger so a future session does not hunt for it.
- Collapsible sections (noted in the task) were not needed - the flex layout keeps
  all three visible without them. Left out deliberately.
