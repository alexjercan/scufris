# FastAPI backend: serve dashboard + /api/stats, scufris CLI launch

- STATUS: CLOSED
- PRIORITY: 28
- TAGS: feature, backlog, dashboard

## Goal

Build the FastAPI backend that serves the built dashboard and exposes
`GET /api/stats` (the `HostStats` snapshot), wired to launch via the `scufris`
CLI with uvicorn. Add `nodejs` to the nix dev shell so the frontend can build.
This is the backend half of the first running dashboard slice.

## Steps

- [x] Add `pkgs.nodejs` to `devShells.default` packages in `flake.nix` so `npm`
      is on PATH under `nix develop`.
- [x] Add settings via pydantic-settings in `scufris/config.py` (host, port,
      poll defaults, `web_dist` path); update `.env.example`.
- [x] Build the FastAPI app in `scufris/app.py`: `GET /api/stats` returns the
      `HostStats` from a module-level `PsutilCollector`; mount `web/dist` via
      `StaticFiles(html=True)` at `/` when the directory exists (log a hint when
      it does not, so the API still runs before the frontend is built).
- [x] Wire `scufris/__main__.py` `main()` to launch `uvicorn.run(app, host, port)`
      from settings.
- [x] Tests in `tests/test_app.py`: `TestClient` `GET /api/stats` against a faked
      collector (assert JSON shape); assert `/` serves the index when a temp
      `web/dist/index.html` exists.
- [x] Run `ruff check .`, `mypy .`, `pytest` green; confirm `scufris` boots and
      `GET /api/stats` responds (curl).

## Definition of Done

- Running `scufris` starts a uvicorn server; `GET /api/stats` returns the host
  stats JSON, and `/` serves the dashboard when `web/dist` is present.
- `nodejs` is available in the dev shell; `.env.example` documents the settings.
- ruff, mypy and pytest are green.

## Notes

- Spike: tasks/20260719-153034/SPIKE.md.
- FastAPI app: mount `web/dist` via `StaticFiles` at `/`, and add `GET /api/stats`
  returning the `HostStats` model from the collector (tatr 20260719-154420).
- `scufris` console entry (`scufris/__main__.py`) launches uvicorn to serve the
  app (host/port from pydantic-settings + `.env`; update `.env.example`).
- Add `pkgs.nodejs` to `devShells.default` packages in `flake.nix` so `npm` is
  available under `nix develop` for the `web/` build.
- Decide how the built frontend is located: for local dev, serve `web/dist` from
  a known path relative to the repo/root. Nix packaging of `web/dist` into
  `nix build .#scufris` is an open question in the spike - a follow-up can bundle
  it; the first slice just needs to run under `nix develop`.
- Harness-first test: hit `GET /api/stats` with a FastAPI TestClient against a
  faked collector and assert the JSON shape; a route test that `/` serves the
  index when `web/dist` exists.
- Depends on the collector (tatr 20260719-154420) for the stats model; pairs with
  the frontend task (tatr 20260719-154539).

## Implementation

- `scufris/config.py`: pydantic-settings `Settings` (SCUFRIS_ prefix, `.env`):
  host, port, `web_dist` (defaults to `<repo>/web/dist` via `__file__`),
  `poll_seconds`. `.env.example` documents them.
- `scufris/app.py`: `create_app(collector, settings)` factory (collector injected
  for tests). `GET /api/stats` returns `HostStats`; `GET /api/config` exposes the
  poll interval to the client. `web/dist` is mounted at `/` via `StaticFiles`
  only when present (logs a build hint otherwise), and mounted AFTER the API
  routes so they win. `main()` runs uvicorn from settings.
- `scufris/__main__.py` now delegates to `scufris.app.main` (console script +
  `python -m scufris`).
- `flake.nix`: added `pkgs.nodejs` to the dev shell for the `web/` build.
- Tests (`tests/test_app.py`, shared fixtures in `tests/conftest.py`): stats
  JSON shape, config endpoint, index served when dist exists, and API-wins-over-
  static-mount. Boot smoke: launched `python -m scufris`, curled `/api/stats`
  and got real host metrics (nixos, kernel 6.18.37). ruff+mypy+pytest green.
