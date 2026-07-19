# Scaffold web/ TypeScript + webpack + Tailwind dashboard page

- STATUS: OPEN
- PRIORITY: 25
- TAGS: feature,backlog,dashboard,ui

## Goal

Scaffold the `web/` TypeScript + webpack + Tailwind single-page dashboard: the
build tooling plus one styled page that fetches host stats from `/api/stats` and
renders them as read-only stat cards. This is the frontend half of the first
running dashboard slice.

## Steps

- [ ] Create the `web/` project: `package.json` (scripts: build, serve, format,
      lint, ci), `tsconfig.json`, `webpack.config.js` (ONE entry `src/main.ts`,
      `HtmlWebpackPlugin` for `src/index.html`, css chain style-loader ->
      css-loader -> postcss-loader, output to `web/dist`), `tailwind.config.js`,
      `postcss.config.js`, ESLint + Prettier - copied down from
      `~/personal/nova-protocol/web` and simplified to one page.
- [ ] Dev server: port + `historyApiFallback`, and a proxy for `/api` -> the
      uvicorn backend port (mirrors nova's `/play` -> trunk proxy).
- [ ] `src/index.html` with a root element; `src/style.css` with Tailwind
      directives + CSS variables for a Scufris dark "scuffed Jarvis" theme.
- [ ] `src/main.ts`: fetch `/api/stats`, render the v1 layout (header:
      hostname/os/uptime; grid of cards: CPU overall + per-core bars, Memory,
      Swap, Load average, Disks per mount, Network IO), and poll every N seconds
      to re-render. Type the response to the `HostStats` shape.
- [ ] Add `web/node_modules/` and `web/dist/` to `.gitignore`.
- [ ] `npm install`, `npm run build` produces `web/dist/`; `npm run ci`
      (format:check + lint + build) green. Verify the page renders live stats
      against the running backend.

## Definition of Done

- `npm run build` produces `web/dist/` with `index.html` + the bundle.
- Served by the backend, the page shows live host stats in styled cards and
  refreshes on the poll interval.
- `npm run ci` is green; `node_modules/` and `dist/` are gitignored.

## Notes

- Spike: tasks/20260719-153034/SPIKE.md (recommends a simplified single-entry
  version of the nova-protocol `web/` build pattern).
- Reference build to copy down and simplify: /home/alex/personal/nova-protocol/web
  (package.json, webpack.config.js, tsconfig.json, tailwind.config.js,
  postcss.config.js, eslint/prettier, `npm run ci`). Reduce to ONE entry point.
- Layout v1: header (hostname / OS / uptime) + a grid of stat cards - CPU
  (overall + per-core bars), Memory, Swap, Load average, Disks (row per mount),
  Network IO. Read-only; no RUN buttons.
- Styling: Tailwind for layout + a small custom `style.css` with CSS variables
  for a Scufris "scuffed Jarvis" dark theme (the user wants a custom style/CSS).
- Data: `main.ts` polls `/api/stats` every N seconds and re-renders cards.
  Polling is the reversible seam (SSE/WebSocket later). Consumes the `HostStats`
  shape from tatr 20260719-154420.
- Dev loop: `npm run serve` (webpack dev server) with a proxy for `/api` -> the
  uvicorn backend port, mirroring nova's `/play` -> trunk proxy.
- Build output to `web/dist/`, served by the backend (tatr 20260719-154544).
- Pairs with the backend task; together with the collector they are the first
  running slice.
