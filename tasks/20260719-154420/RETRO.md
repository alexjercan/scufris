# Retro: Build psutil-backed host metrics collector

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The spike had already picked psutil behind a fakeable seam, so implementation
  was mechanical: models -> protocol -> collector -> tests, no design churn.
- The `Collector` protocol paid off immediately: the fake-collector test doubles
  as the fixture the FastAPI backend task will reuse, so no rework there.
- Adding a compiled dep (psutil) through uv2nix Just Worked: `uv add` +
  `nix develop --command` rebuilt the venv from the wheel with no source build.

## What went wrong / friction

- First mypy run failed: `scufris/` had no `__init__.py`, so mypy saw
  `metrics.py` as both `metrics` and `scufris.metrics`. Fixed by adding
  `scufris/__init__.py`. Worth knowing for every future module in this package.
- New dependencies are NOT visible to the already-active dev shell: the session
  runs against a fixed nix-store venv, so a bare `pytest` would not see psutil.
  Had to run checks via `nix develop --command ...` to rebuild the venv with the
  new lock. This is the standard loop for any dep change here.

## Lessons

- `metrics-module-needs-package-init`: a new `scufris/` module needs the package
  to have `__init__.py` or mypy errors on duplicate module paths. (Now present.)
- `dep-change-needs-nix-develop-rebuild`: after `uv add`, run tests/type-checks
  through `nix develop --command` (or re-enter the shell) so the uv2nix venv
  rebuilds with the new lock; the active shell's venv is frozen.

## Follow-ups

- Optional LOW hardening from REVIEW.md (guard `net_io_counters()` returning
  None) - only if it ever matters on a real host; not filed as a task.
