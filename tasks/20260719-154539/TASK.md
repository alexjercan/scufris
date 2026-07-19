# Scaffold web/ TypeScript + webpack + Tailwind dashboard page

- STATUS: OPEN
- PRIORITY: 25
- TAGS: feature,backlog,dashboard,ui

## Goal

Scaffold the `web/` TypeScript + webpack + Tailwind single-page dashboard: the
build tooling plus one styled page that fetches host stats from `/api/stats` and
renders them as read-only stat cards. This is the frontend half of the first
running dashboard slice.

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
