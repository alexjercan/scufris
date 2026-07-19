# FastAPI backend: serve dashboard + /api/stats, scufris CLI launch

- STATUS: OPEN
- PRIORITY: 28
- TAGS: feature,backlog,dashboard

## Goal

Build the FastAPI backend that serves the built dashboard and exposes
`GET /api/stats` (the `HostStats` snapshot), wired to launch via the `scufris`
CLI with uvicorn. Add `nodejs` to the nix dev shell so the frontend can build.
This is the backend half of the first running dashboard slice.

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
