# Multi-page app: agent landing page + separate stats page

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,ui,dashboard

## Goal

Restructure the frontend into multiple pages: the LANDING page is the agent chat,
with the stats dashboard on a SEPARATE page, linked from the main page (nav /
header). Leave room to add more pages later.

## Notes

- Today `web/` is a single-entry SPA (`src/main.ts` -> `dashboard.ts`) that
  renders stats + a chat panel on one page. This task splits them: agent chat as
  the default/landing route, stats as its own route/page, with navigation
  between them.
- Build: move webpack to multi-entry with an `HtmlWebpackPlugin` per page and
  `historyApiFallback` for clean routes - mirror the nova-protocol `web/` pattern
  (multiple entries + per-page HTML + dev-server rewrites). Split `dashboard.ts`
  so the stats rendering and the chat live in separate page modules over shared
  helpers.
- Backend: `StaticFiles(html=True)` already serves the built bundle; ensure the
  server serves each page (and its route) correctly.
- No spike (user's call): this is a defined restructure. `/plan` it into steps
  when picked up.
- Sequencing: the richer stats (tatr 20260719-180507) and the enriched agent page
  (tatr 20260719-180528) will land ON these pages, so consider doing this
  restructure around the same time / first. "Then we think of what else should go
  in here" - keep the nav extensible for future pages.
- User ask (2026-07-19): landing = agent, stats on a different page linked from
  main; more pages TBD.
