# Retro: FastAPI backend - serve dashboard + /api/stats, scufris CLI

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The collector's `Collector` protocol made the backend trivial to test: inject
  a fake, no psutil in the API tests. The seam paid off exactly as the metrics
  spike predicted.
- Route precedence (API before the `/` static mount) was pinned with a dedicated
  test, so a future reorder can't silently shadow `/api/*`.
- The app is independently shippable: a missing `web/dist` logs a hint and serves
  API-only, so the backend task landed without waiting on the frontend.
- Boot smoke against a real host (curl `/api/stats` -> real nixos metrics) caught
  nothing broken but gave real end-to-end confidence, per harness-first.

## What went wrong / friction

- mypy rejected `from .conftest import FakeCollector` because `tests/` is not a
  package. Rather than add `tests/__init__.py`, typed the fixture params as the
  `Collector` protocol and dropped the class import entirely - cleaner, and the
  fixture (not the concrete class) is the reuse surface.
- Two ruff-format round-trips: wrote lines ruff then reformatted. Minor, but
  running `ruff format .` (not `--check`) as the first step of the check loop
  would save a cycle.

## Lessons

- `type-test-fixtures-by-protocol`: annotate injected test doubles by the
  protocol they satisfy (`Collector`), not the concrete fake class, to avoid
  cross-test imports that mypy can't resolve without making `tests/` a package.
- `web_dist-via-__file__-is-dev-only`: the `<repo>/web/dist` default resolves
  correctly only for the editable dev install; packaging the built assets into
  the wheel/nix closure is still open (owned by a future task).

## Follow-ups

- Nix packaging of `web/dist` into `nix build .#scufris` (spike open question) -
  becomes a real task once the frontend build exists and we want `nix run` to
  serve the dashboard without a separate `npm run build`.
