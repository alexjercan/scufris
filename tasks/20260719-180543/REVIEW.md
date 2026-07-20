# Review: Multi-page app (agent landing + stats page)

## Round 1 - 20260719

Scope: `web/src` split (`common.ts`, `stats-view.ts`, `agent-view.ts`, `nav.ts`,
`agent.ts`, `stats.ts`, `index.html`, `stats.html`), `web/webpack.config.js`,
`web/src/style.css` (nav), `tests/test_app.py`, `stats-view.test.ts`.

### Correctness

- Proven live: `/` serves the agent chat page (200, chat + nav), `/stats/` serves
  the stats page (200, cards + nav), `agent.js`/`stats.js` both 200, `/api/stats`
  200, and `POST /api/chat` still replies. Each page pulls only its own bundle.
- Clean module boundary: shared `common.ts`; `stats-view.ts` and `agent-view.ts`
  stay side-effect-free (start functions exported, not called), so the jsdom
  tests still import render fns without kicking off fetch/timers - the
  `side-effect-free-module-for-jsdom-tests` lesson is preserved through the
  refactor. Thin entries (`agent.ts`/`stats.ts`) hold the side effects.
- Backend needed no change: `StaticFiles(html=True)` already resolves `/` ->
  index.html and `/stats/` -> stats/index.html; the new
  `test_stats_page_served_at_subpath` pins it.
- The escaping hardening carried over intact (mountpoint/host escaped in
  `stats-view`; the injection tests moved with it). ruff+mypy+pytest and
  `npm run ci` (4 jsdom tests) green; `node_modules` gitignored.
- Nav is a plain `<nav>` with `.nav__link`s and an `initNav()` active-marker,
  driven off `location.pathname` - trivial to extend with more pages, matching
  the task's "keep the nav extensible" intent.

### Observations (non-blocking)

- MINOR: `/stats` without a trailing slash is not served (only `/stats/`); the nav
  links use `/stats/`, so users never hit it, but a stray `/stats` would 404. A
  redirect could be added if it ever matters.
- MINOR: the two HTML templates duplicate the header/nav markup. Fine for two small
  pages; if pages proliferate, a partials plugin (as nova-protocol uses) would
  de-duplicate. Deferred deliberately.
- NIT: header/nav is shared by copy, not a component; acceptable at this size.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: the build produces an agent landing page
at `/` and a stats page at `/stats/`, each loading its own bundle, with a shared
extensible nav; the backend serves both; chat and live stats both work
(serve-verified); checks green. MINOR items are appropriate to defer.
