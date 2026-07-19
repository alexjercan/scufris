# Retro: btop-style sparklines on the stats cards

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Re-cutting the plan paid off. The task's original note called for a backend
  `BackgroundSampler` + ring buffer + `GET /api/history`. Stepping back, "a
  minigraph like btop" is a live rolling window that fills from start, not
  persisted history - so a client-side ring buffer over the poll the page
  already runs delivered the exact same visible result with ZERO backend
  surface (no lifespan task, no memory-bound endpoint, no new tests on the
  Python side). The whole feature landed in one frontend file + CSS + tests.
- The settled frontend patterns did all the heavy lifting again: `sparkline`
  is a pure exported function (side-effect-free-module-for-jsdom-tests), the
  history Map got a `_resetStatsHistory` test hook
  (persistent-ui-state-needs-a-test-reset-hook), and it was unit-tested in
  jsdom without a browser. First-try green on both suites.
- Inline SVG (area polygon + polyline on a 100x30 viewBox, stretched with
  `preserveAspectRatio=none` + `vector-effect: non-scaling-stroke`) gave a
  crisp, theme-coloured graph that scales to any card width with no canvas,
  no per-frame redraw, and no dependency.

## What went wrong / friction

- The sprout worktree had no `node_modules`; a bare `npm run ci` would have
  failed. Symlinking the main checkout's `node_modules` into the worktree was
  instant and correct (webpack/vitest resolve through the link fine). Worth
  remembering for every frontend task that sprouts a fresh worktree.
- `.gitignore` has `node_modules/` (trailing slash = directories only), so the
  symlink showed as untracked rather than ignored. Handled by staging only the
  three real source files explicitly - never `git add -A` in the worktree.

## Lessons

- `client-side-rolling-window-beats-backend-history-for-live-graphs`: for a
  btop-style live graph, accumulate samples client-side over the poll the page
  already runs; a backend sampler/`/api/history` only earns its complexity when
  cross-reload or cross-client persistence is an actual requirement.
- `symlink-node_modules-into-fresh-worktrees`: a sprouted worktree has no
  `node_modules`; `ln -s <main>/web/node_modules <wt>/web/node_modules` beats a
  reinstall. The `node_modules/` (dir-only) ignore rule misses the symlink, so
  stage source files explicitly.

## Follow-ups

- Optional visual polish (user's eyeball pass): GPU card places the util
  sparkline between the util and VRAM bars - could group both bars then graph.
- The deferred backend sampler + `/api/history` remains a clean, still-unbuilt
  option if persistent/shared history is ever wanted. No open task for it.
