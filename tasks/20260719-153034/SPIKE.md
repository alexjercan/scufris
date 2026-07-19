# Spike: Dashboard app style and layout

- DATE: 20260719-153034
- STATUS: RECOMMENDED
- TAGS: spike, backlog, dashboard, ui

## Question

What should the Scufris dashboard be built as, and what is the layout for a
first, simple, read-only dashboard that shows host stats and runs end to end as
a starting point?

## Context

User direction (2026-07-19) narrowed this before the spike: keep it simple and
build a **CLI + webpack-based UI using TypeScript and Tailwind**, mirroring how
`nova-protocol/web` is built, but simpler. The dashboard is **read-only**: the
FastAPI backend samples host stats (psutil, see [[20260719-153045]]) and other
read-only host info (e.g. the list of `tatr` tasks) and serves them over GET;
there are no "RUN" buttons. Tool-*running* is the agent's job (chat), scoped in
[[20260719-153050]]. First goal is explicitly a simple dashboard with a custom
style/CSS that just runs - a vertical slice, not the full product.

The backend stack already exists in `pyproject.toml`: FastAPI + Uvicorn, pydantic.
So the natural shape is FastAPI serving a JSON stats API plus the static
frontend bundle.

## Options considered

- **Web SPA behind FastAPI (RECOMMENDED)** - a single-page TypeScript app built
  with webpack + Tailwind (the nova-protocol `web/` pattern, reduced to one
  entry point), output to `dist/`, served by FastAPI `StaticFiles` alongside a
  `/api/*` JSON surface. Live updates by the page polling `/api/stats` on an
  interval (simplest; upgrade to SSE/WebSocket later behind the same seam). Pros:
  matches the user's chosen stack and an existing, proven build pattern; reuses
  the FastAPI backend; real charts/styling possible; the eventual chat panel and
  read-only info panels drop into the same SPA. Cons: a JS build to maintain
  (mitigated by copying nova's config).
- **Rich/Textual TUI** - terminal dashboard, no browser. Pros: light, no JS.
  Cons: the user explicitly wants a webpack/TS/Tailwind web UI, and a browser
  chat panel is a better home for the agent. Rejected on direction.
- **Native/desktop shell** (pywebview/Tauri) - a window around the web UI. Pros:
  app feel. Cons: extra nix packaging now, no benefit for a single local host
  reachable at localhost. Defer; the SPA can be wrapped later if wanted.
- **Do nothing yet / JSON API only** - ship `/api/stats` and defer the UI. Costs
  the visible starting point the user asked for. Rejected as the end state, but
  the API is built first inside the recommended option anyway.

## Recommendation

Build a **single-page TypeScript + webpack + Tailwind dashboard, served by the
FastAPI backend**, following the nova-protocol `web/` scaffolding but trimmed to
one entry point.

Concrete shape:

- `web/` project (sibling to `scufris/`): `package.json`, `webpack.config.js`,
  `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`, ESLint + Prettier,
  and an `npm run ci` (format:check + lint + build) - all copied down from nova
  and simplified. One entry `src/main.ts` importing `src/style.css`, one
  `src/index.html` with a root element, build to `web/dist/`.
- Styling: Tailwind for layout + a small `style.css` with custom CSS variables
  for a Scufris theme (a dark "scuffed Jarvis" palette), matching the user's
  "custom style and CSS" ask. Keep markup semantic with a few component classes
  rather than a wall of utilities.
- Backend: FastAPI app mounts `web/dist` as static files at `/`, and exposes
  `GET /api/stats` returning the `HostStats` model (from the metrics collector).
  The `scufris` console entry launches uvicorn to serve it.
- Live update: `main.ts` fetches `/api/stats` every N seconds and re-renders
  stat cards (CPU, memory, swap, disks, load, uptime, network). Polling is the
  single reversible lever - swap to SSE/WebSocket later without touching the UI
  components.
- Dev loop: `npm run serve` (webpack dev server) with `historyApiFallback` and a
  proxy for `/api` -> the uvicorn port (mirrors nova's `/play` proxy to trunk),
  so frontend and backend hot-reload side by side. One-shot: `npm run build`
  then run `scufris` to serve the combined output (closest to production).

First-screen layout (v1): a header (hostname / OS / uptime), then a grid of stat
cards - CPU (overall + per-core bars), Memory, Swap, Load average, Disks (one
row per mount), Network IO. A read-only "tatr tasks" panel and the chat panel
are later additions to the same SPA, not part of the first slice.

This beats the runners-up because it is exactly the stack the user chose, reuses
a build pattern already working in a sibling repo, and gives a running, styled
dashboard as the first slice while leaving clean seams for the agent chat panel
and richer transports.

## Open questions

- Where the built frontend lives in the nix package: `nix build .#scufris` should
  include `web/dist`. Decide whether the Python package bundles the built assets
  (build the JS first, ship as package data) or the flake builds `web/` as a
  separate derivation and the app finds it via a path/env. Resolve during
  implementation; for local dev, serving `web/dist` from a known path is enough.
- Node/webpack toolchain in the dev shell: add `nodejs` (and keep `npm`) to the
  flake `devShells.default` packages so `npm` is available under `nix develop`.
- Exact poll interval and whether per-core CPU is in v1 or a fast follow.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260719-154539: scaffold the `web/` TypeScript + webpack + Tailwind
  single-page dashboard (build tooling + one styled page that renders host
  stats from `/api/stats`).
- tatr 20260719-154544: FastAPI backend that serves the built dashboard and
  exposes `GET /api/stats`, wired to launch via the `scufris` CLI (uvicorn);
  add `nodejs` to the nix dev shell.

Together with the metrics collector (tatr 20260719-154420, from
[[20260719-153045]]) these three form the first running vertical slice: a simple
dashboard that shows real host stats.

## Fix record

(Appended by each implementing task as it lands.)
