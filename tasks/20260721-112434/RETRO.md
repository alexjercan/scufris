# Retro: F2 render agents as cards + friendly labels + card->page nav

- TASK: 20260721-112434
- BRANCH: feature/agent-cards
- REVIEW ROUNDS: 1 (out-of-context APPROVE with 2 MINOR + 3 NIT, addressed)

## What went well

- The pure `renderX(root, data, actions)` + injected-actions seam made the
  rewrite cheap: swapping the in-page detail panel for card->page navigation
  was a signature change (`select/run` -> `open`) plus a new `statuses` map, all
  driven by jsdom tests with no fetch. The seam paid off exactly as designed.
- Reusing the Stats `.cards`/`.card`/`.row` shape gave the cards a consistent
  look for free - no new grid CSS, just an `.agents__card` clickable modifier.
- The out-of-context reviewer caught a real keyboard-interaction bug (Enter on
  the focused delete button both deletes and navigates) that the mouse-path
  test masked - the click handler's `stopPropagation` does nothing for a
  separate `keydown` event.

## What went wrong

- R1.1 (keyboard-delete-navigates): I guarded the mouse path
  (`ev.stopPropagation()` on the delete click) but not the keyboard path. Root
  cause: I reasoned about "delete must not navigate" only through the click
  event I had just written, and did not enumerate the OTHER activation path a
  `role=button tabindex=0` card opens up. A clickable container with an inner
  button has two bubbling channels (click, keydown); guarding one is a
  half-fix.
- R1.2 (dead CSS): the rewrite removed the consumers of `.agents__item`,
  `.agents__name`, `.agents__status`, `.agents__events` etc. but I left the
  rules behind. Root cause: I added the new `.agents__card` rules without
  sweeping the sibling rules the deletion orphaned - a symbol-removal sweep
  that stopped at the TS/HTML and never reached the stylesheet.
- R1.3/R1.4 (weak tests): a `toContain("2")` turns assertion was
  substring-lucky (the tokens line already contains "2"), and the XSS test
  asserted description-escaping the card never renders. Both pass whether or
  not the mechanism works - exactly the "would it fail if deleted?" smell.

## What to improve next time

- When a widget is a clickable container wrapping an interactive child,
  enumerate BOTH activation paths (pointer click and keyboard Enter/Space) and
  guard/verify each - a `stopPropagation` on click is not symmetric with
  keydown. Add a test for the keyboard path, not just the mouse path.
- Extend the removal sweep to stylesheets: when a render rewrite drops DOM
  structure, grep the CSS for the classes it stopped emitting and delete the
  orphans in the same diff (the work skill's "grep for everything that observes
  the mechanism" includes `.css`).
- For a "shows X" test, assert the specific rendered node's value (query the
  row, compare the value span), and pick fixture values that cannot appear
  elsewhere on the card - a bare `toContain` on a common digit proves nothing.

## Action items

- [x] All review findings addressed in round 1 (no follow-up tasks needed).
- Carried forward: F3 (the next task) makes the `/agents/<id>` detail page's
  settings editable - it will reuse this card page's `open` target.
