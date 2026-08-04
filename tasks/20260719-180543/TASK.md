# Multi-page app: agent landing page + separate stats page

- PRIORITY: 0
- TAGS: feature, backlog, ui, dashboard
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Restructure the frontend into multiple pages: the LANDING page is the agent chat,
with the stats dashboard on a SEPARATE page, linked from the main page (nav /
header). Leave room to add more pages later.

## Steps

- [x] Split the frontend modules (keeping the side-effect-free-for-tests rule):
      `src/common.ts` (shared `el`, `escapeHtml`, `fetchJson`, `HostStats`/config
      types), `src/stats-view.ts` (stats render fns + `startStats()`, exported, no
      side effects), `src/agent-view.ts` (chat logic + `startAgent()`, exported,
      no side effects), and a `src/nav.ts` to mark the active nav link. Thin
      entries `src/stats.ts` and `src/agent.ts` import the css + view + nav and
      call start. Remove the old `main.ts`/`dashboard.ts`.
- [x] Two HTML templates with a shared header/nav (Agent | Stats): `src/index.html`
      = agent landing (nav + chat markup), `src/stats.html` = stats page (nav +
      host-summary + cards + status). Nav links `/` and `/stats/`.
- [x] webpack: two entries (`agent`, `stats`); `HtmlWebpackPlugin` per page
      (`index.html` <- agent chunk, `stats/index.html` <- stats chunk);
      `historyApiFallback` rewrite for `/stats`. Build to `web/dist`.
- [x] Backend: confirm `StaticFiles(html=True)` serves `/` (agent) and `/stats/`
      (stats); add a backend test for the stats route.
- [x] Tests: repoint the jsdom render tests to `stats-view` (keep the escaping +
      injection cases); `npm run ci` green.
- [x] LIVE serve smoke on this host: `/` serves the agent chat page, `/stats/`
      serves the stats page, both bundles 200, nav present. `ruff`/`mypy`/`pytest`
      + `npm run ci` green.

## Definition of Done

- The build produces an agent landing page at `/` and a stats page at `/stats/`,
  each loading only its own bundle, with a shared nav linking them.
- The backend serves both routes; chat works on the agent page and live stats
  work on the stats page (serve smoke). Tests + `npm run ci` green; the nav is
  easy to extend with more pages later.

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

## Implementation

- Split `web/src` into two page entries over shared modules: `common.ts`
  (types + `el`/`escapeHtml`/`fetchJson`/`loadConfig`), `stats-view.ts` (stats
  render + `startStats`, side-effect-free), `agent-view.ts` (chat + `startAgent`,
  side-effect-free), `nav.ts` (active-link). Thin entries `agent.ts` and
  `stats.ts` import css + nav + view and call start. Removed `main.ts` /
  `dashboard.ts`.
- Two templates with a shared header/nav: `index.html` (agent landing, chat) and
  `stats.html` (host-summary + cards). Nav links `/` and `/stats/`.
- webpack: two entries (`agent`, `stats`), an `HtmlWebpackPlugin` per page
  (`index.html` <- agent, `stats/index.html` <- stats), `historyApiFallback`
  rewrite for `/stats`. Build emits both pages + `agent.js`/`stats.js`.
- Backend needed no change: `StaticFiles(html=True)` serves `/` -> index.html and
  `/stats/` -> stats/index.html; added `test_stats_page_served_at_subpath`.
- Tests: repointed the jsdom render tests to `stats-view`/`common` (escaping +
  injection cases kept). ruff+mypy+pytest and `npm run ci` (incl. 4 jsdom tests)
  green.

### Live verification (DoD)

Serve smoke on this host: `/` -> 200 (agent chat + nav), `/stats/` -> 200 (cards
+ nav), `/agent.js` + `/stats.js` -> 200, `/api/stats` -> 200, and `POST
/api/chat` still replies ("hi") on the agent page. Each page loads only its own
bundle; nav links between them. The nav is a plain list, easy to extend with
future pages.
