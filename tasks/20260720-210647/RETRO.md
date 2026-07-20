# Retro: Projects UI - Projects page

- TASK: 20260720-210647
- BRANCH: feature/projects-page
- REVIEW ROUNDS: 1 (APPROVE, 1 MINOR fixed in-round)

## What went well

- The settings-page patterns transferred wholesale: pure render + injected
  actions seam, sendJson, multipage webpack wiring, escapeHtml discipline. The
  page came together fast and the reviewer found only one race.
- e2e-verified the whole slice end to end (create a project pointing at a real
  dir, see its actual tatr task on the page), not just a green build.

## What went wrong

- One MINOR race the reviewer caught: the lazy tasks fetch wrote a shared
  `tasks` variable without checking the selection was still current, so
  rapidly selecting A then B could render A's tasks under B. Fixed by guarding
  every write/render with `if (selectedId === id)`.

## What to improve next time

- Any async handler that writes shared UI state keyed by a selection must
  re-check the selection is still current before applying - the classic
  last-write-wins race. Guard by the id it was fired for.

## Action items

- No lessons ledger entry (the stale-async-selection race is a well-known
  frontend pattern; captured here). No follow-up code task - P0 is complete.
