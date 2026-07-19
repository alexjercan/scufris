# Review: FastAPI backend - serve dashboard + /api/stats, scufris CLI

## Round 1 - 20260719

Scope: `scufris/config.py`, `scufris/app.py`, `scufris/__main__.py`,
`.env.example`, `flake.nix` (nodejs), `tests/conftest.py`, `tests/test_app.py`.

### Correctness

- `GET /api/stats` returns the injected collector's `HostStats`; verified end to
  end by booting `python -m scufris` and curling a real snapshot (nixos, kernel
  6.18.37, real cpu/mem/swap/disk). Meets the Definition of Done.
- Route ordering is correct and pinned: the `StaticFiles` mount at `/` is added
  AFTER the API routes, so `/api/*` resolves even with the bundle mounted
  (`test_api_wins_over_static_mount`).
- The app runs standalone before the frontend exists: a missing `web_dist` logs
  a build hint instead of failing, so this task is independently shippable.
- Collector is injected via `create_app`, so tests never touch psutil; the
  shared `fake_collector` fixture lives in `tests/conftest.py`.

### Observations (non-blocking)

- LOW: `web_dist` defaults to `<repo>/web/dist` via `__file__`, which is correct
  for the editable dev install but not for a packaged wheel. The spike already
  flagged nix packaging of the built assets as an open question; a follow-up owns
  it. Not blocking the first slice, which runs under `nix develop`.
- LOW: `TestClient` emits a StarletteDeprecationWarning about httpx. Cosmetic,
  upstream; no action now.
- NOTE: `get_stats` samples synchronously in the request path. Fine at the
  dashboard's poll rate; if it ever gets heavy, the spike's background-sampler
  option is the documented next step.

### Verdict

APPROVE. The backend meets its Definition of Done, is verified booting against
real host stats, keeps the API/static precedence pinned by a test, and the
checks (ruff, ruff format, mypy, pytest) are green. LOW items are future work.
