# F1: SPA dynamic routing + fallback + the /agents/<id> agent-detail page shell

- PRIORITY: 44
- TAGS: agents, frontend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Introduce real per-agent routing: a FastAPI catch-all that serves the SPA shell
for `/agents/<id>` (and `/agents/<id>/settings`) when the path is not a static
asset, so client-side routing works; add the webpack `agent-detail` entry +
`historyApiFallback` for `/^\/agents\//`. This is the structural gate for the
per-agent page (F3) and chat (F4).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 1; recommendation F1). The MPA/no-fallback gap is the
  single biggest structural blocker.
- No hard dep, but pairs with F2/F3.

## Steps

- [x] app.py: serve an agent-detail SPA shell for `/agents/{id}` and
      `/agents/{id}/{rest:path}` (e.g. .../settings) via routes registered BEFORE
      the static mount, returning the built `agent-detail.html` (FileResponse,
      no-cache; 404 when the frontend is not built). `include_in_schema=False`
      (page, not API). `/agents/` (list) stays on the static index; `/api/...`
      unaffected.
- [x] webpack: `agent-detail` entry + HtmlWebpackPlugin (filename
      `agent-detail.html`, chunks ["agent-detail"]); devServer
      `historyApiFallback` rewrite `/^\/agents\/.+/ -> /agent-detail.html` placed
      BEFORE the `/^\/agents/ -> /agents/index.html` list rewrite.
- [x] `agent-detail.html` shell (#agent-detail) + `agent-detail.ts` thin entry
      (`initNav(); void startAgentDetail();`).
- [x] `agent-detail-view.ts`: `agentIdFromPath(pathname)` (parse `/agents/<id>`
      or `/agents/<id>/settings`); a PURE `renderAgentDetail(root, agent, project,
      status)` (read-only for F1: name, back link, project/backend/model/
      description/mode + status); `startAgentDetail` fetches the agent + project +
      status and polls. (Editable settings = F3; chat = F4.)
- [x] Tests: backend - `/agents/<id>` serves the shell (200 html) when a dist
      with agent-detail.html exists, 404 when not, and `/api/agents/<id>` is
      unaffected; frontend - `agentIdFromPath` parsing + `renderAgentDetail`
      renders the fields + a hostile string escaped.
- [x] Full suite + npm run ci green; close-out.

## Definition of Done

- `/agents/<id>` serves the agent-detail shell; `/api/agents/<id>` still returns
  JSON (test: `agent_detail_page_serves_shell`).
- `agentIdFromPath` parses `/agents/<id>` and `/agents/<id>/settings`
  (test: `agent_id_from_path`).
- `renderAgentDetail` shows the agent's read-only fields + a back link
  (test: `agent_detail_renders`).
- Full suite + npm run ci green.

## Close-out

What changed:
- app.py: `GET /agents/{id}` and `GET /agents/{id}/{rest:path}` serve the built
  `agent-detail.html` SPA shell (FileResponse, no-cache; 404 until built;
  include_in_schema=False). Registered before the static mount, so `/agents/`
  (list) stays on the static index and `/api/...` is unaffected. Starlette's
  non-empty path converter means `/agents/` doesn't match the `{id}` route.
- webpack: `agent-detail` entry + HtmlWebpackPlugin (`agent-detail.html`);
  devServer rewrite `/^\/agents\/.+/ -> /agent-detail.html` BEFORE the list
  rewrite.
- `agent-detail.html` shell + `agent-detail.ts` entry.
- `agent-detail-view.ts`: `agentIdFromPath` (parses `/agents/<id>[/settings]`),
  pure `renderAgentDetail` (read-only detail: name/back-link/project/backend/
  model/description/mode + status), `startAgentDetail` (fetch + poll). Editable
  settings = F3, chat = F4.
- Tests: backend (shell served for `/agents/<id>` and `/settings`, list not
  shadowed, `/api/agents/<id>` unaffected, 404 without a build); frontend
  (agentIdFromPath cases, renderAgentDetail fields + not-started + hostile
  escape).

Verification:
- 258 backend (+2) + 141 frontend (+6) tests, ruff + mypy clean, npm run ci green.
- E2E through the real backend + built bundle: `/agents/` -> list, `/agents/<id>`
  and `/agents/<id>/settings` -> the detail shell, `/api/agents` -> JSON.

Self-reflection: the one design care was route ordering + Starlette's non-empty
path segment (so `/agents/{id}` doesn't swallow `/agents/`). Verified with an
explicit test AND a live serve, since routing is exactly the kind of thing a
green unit test can miss end to end.
