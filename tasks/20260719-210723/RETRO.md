# Retro: sparkline labels/tooltips + GPU VRAM bar placement

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Small, well-scoped follow-up to the sparkline feature; both user asks (label +
  tooltip, VRAM bar placement) landed in one cycle on one file + CSS + tests.
- The `symlink-node_modules-into-fresh-worktrees` lesson from the previous task
  paid off immediately - linked deps into the fresh worktree and `npm run ci` ran
  first try, no reinstall.
- SVG `<title>` was the right primitive for the tooltip: native hover + a11y name
  for free, no JS handlers, no positioning math. Keeping `sparkline` returning
  the bare svg and adding a separate `labeledSpark` wrapper meant the four
  existing `sparkline` unit tests kept passing untouched (title is an optional
  4th arg defaulting to "").
- Testing the VRAM fix by asserting the vram row's `nextElementSibling` is a
  `.bar` pins the *ordering* the user asked for, not just presence - a
  presence-only assertion would have passed even with the bar still above.

## What went wrong / friction

- Prettier failed `format:check` on the new test (a one-line arrow that wanted
  wrapping). `npx prettier --write` on the one file fixed it; re-ran ci green.
  Reflex for future frontend tasks: run `prettier --write` before `npm run ci`,
  not after it complains.

## Lessons

- (No new ledger entry - reused `side-effect-free-module-for-jsdom-tests`,
  `symlink-node_modules-into-fresh-worktrees`, and the escape lesson. The
  prettier-write reflex is a minor habit, not worth a ledger slot yet; if it
  recurs, promote it.)

## Follow-ups

- None. Optional: the corner caption slightly overlaps a high graph line; the
  semi-opaque chip background keeps it legible. User eyeball will decide if it
  needs more.
