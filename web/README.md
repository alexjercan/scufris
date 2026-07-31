# `web/` - the dashboard frontend

A multi-page TypeScript app built with webpack and Tailwind. No framework: each
page loads only its own entry bundle, talks to the FastAPI backend over
`/api/...`, and streams live updates over SSE.

The built output lands in `web/dist`, which the Python app serves at `/` via
`SCUFRIS_WEB_DIST`. The Python wheel deliberately does NOT ship `dist` - the nix
`packages.scufris-web` derivation provides it, and the service module wires the two
together.

## Setup

```sh
cd web
npm ci            # deps, once per checkout (node comes from `nix develop`)
npm run build     # build web/dist, which the app then serves at /
```

For frontend work, run the backend and the dev server side by side.
`npm run serve` reads the repo-root `.env` so it proxies `/api` to the same port
the Python app binds:

```sh
scufris serve     # in one shell
npm run serve     # in another: webpack-dev-server with live reload
```

## Scripts

| Command | What it does |
|---|---|
| `npm run build` | webpack build into `dist/` |
| `npm run serve` | webpack-dev-server with an `/api` proxy to the backend |
| `npm run format` / `format:check` | prettier over `src/` and the config files |
| `npm run lint` / `lint:fix` | eslint over `src/**/*.ts` |
| `npm test` | vitest (jsdom) |
| `npm run ci` | **the gate**: format:check + lint + test + build |

`npm run ci` is what has to be green before landing a frontend change. It is not
part of `nix flake check`, so run it yourself when you touch anything here.

## The pages

| URL | Entry | What it is |
|---|---|---|
| `/` | `agent.ts` | the landing orchestrator chat |
| `/stats/` | `stats.ts` | the live host dashboard (CPU, memory, disk, network, processes, temperatures) |
| `/host/` | `host.ts` | the host approval queue and the audit page |
| `/agents/` | `agents.ts` | the agent cards, plus `agent-detail.html` for one agent's chat at `/agents/<id>` |
| `/projects/` | `projects.ts` | project records and discovery, plus `project-detail.html` |
| `/settings/` | `settings.ts` | the runtime-mutable settings surface |
| `/login/` | `login.ts` | the operator login form |

`_header.html` and `_footer.html` are injected into every page by
`webpack-partials.js`, so the nav lives in one place.

## Conventions

- **View logic is separated from the DOM entry.** Most pages have a thin
  `<page>.ts` that wires up the document and a `<page>-view.ts` holding the logic, which is
  what the `*.test.ts` files exercise under jsdom. That split is why the tests can
  cover rendering without a browser.
- **A page bigger than one file becomes flat siblings, not a directory.** The
  `<page>-view.ts` keeps the page's shape (what renders where, and the poll or
  stream that feeds it) and the sections live beside it under a `<page>-*.ts`
  name, imported directly. There is no barrel and no `index.ts`, so every import
  names the module that owns the symbol:

  | Page | View | Siblings |
  |---|---|---|
  | `/stats/` | `stats-view.ts` | `stats-elements.ts` (bar/card/sparkline primitives and the rolling history), `stats-cards.ts` (the live gauges), `stats-host-cards.ts` (the `/api/host/overview` cards) |
  | `/host/` | `host-view.ts` | `host-format.ts` (the text-only building blocks), `host-actions.ts` (what the page can do), `host-proposal.ts` (the pending cards and every approve control), `host-history.ts` (decided cards and the audit table), `host-checks.ts` (the scheduled checks) |
  | chat | `agent-chat-view.ts` | `agent-chat-types.ts`, `agent-chat-log.ts` (the transcript), `agent-chat-composer.ts` (the slash palette and image attach), `agent-chat-turn.ts` (the streaming turn and its cancel) |

  The wire shapes are siblings too: `stats-types.ts`, `host-types.ts` and
  `agent-types.ts`, with `common.ts` holding only the runtime helpers.
- **Every API call goes through `apiFetch` in `common.ts`**, never bare `fetch`.
  It attaches the CSRF header the server requires on state-changing requests and
  turns a 401 into a trip to `/login/`, mirroring the backend's single enforcement
  middleware. `fetchJson` and `sendJson` are the typed wrappers over it.
- **The `/host/` page must work at phone width.** Approving a change while away
  from the desk is the case it exists for: it shows each waiting proposal with its
  risk class, every command it would run in order, the preview, who asked, and the
  expiry counting down. A one-way action has no ordinary approve button at all -
  the only control that can approve it requires typing the action's name first,
  which is the same rule the server enforces.
- **Tailwind v4** via the postcss plugin; there is no separate config to edit for
  most work.

The API contract these pages consume is listed in
[`../scufris/README.md`](../scufris/README.md#7-the-http-surface).
