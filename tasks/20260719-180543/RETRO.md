# Retro: Multi-page app (agent landing + stats page)

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The prior task's `dashboard.ts` split (side-effect-free view + thin entry) made
  this restructure mostly mechanical: extract shared bits into `common.ts`, split
  the view into `stats-view`/`agent-view`, add two entries. The
  side-effect-free-for-tests discipline carried straight through, so the jsdom
  tests kept working after moving.
- The webpack multi-page pattern (two entries + one `HtmlWebpackPlugin` per page,
  per-page `chunks`) mirrored nova-protocol and built both pages first try.
- The backend needed ZERO changes - `StaticFiles(html=True)` already resolves
  `/stats/` -> `stats/index.html`. Confirmed with a route test rather than
  assumed.
- Serve smoke exercised the real thing: both pages 200, both bundles load, and
  chat still replies - catching any wiring regression a build-only check misses.

## What went wrong / friction

- Nothing notable. The one judgement call was leaving the header/nav duplicated
  across the two HTML templates instead of adding a partials plugin - correct at
  two pages; revisit if pages multiply.

## Lessons

- `webpack-multipage-htmlplugin-per-page`: for a multi-page app, use one
  `entry` per page + one `HtmlWebpackPlugin` per page with explicit `chunks`, and
  a `historyApiFallback` rewrite per sub-route; the FastAPI `StaticFiles(html=True)`
  mount then serves `/` and `/<page>/` with no backend change.

## Follow-ups

- The richer-stats spike (tatr 20260719-180507) and agent-detail spike
  (tatr 20260719-180528) now have their pages to land on.
- A shared header/nav partial (and a `/stats` -> `/stats/` redirect) if the page
  count grows.
