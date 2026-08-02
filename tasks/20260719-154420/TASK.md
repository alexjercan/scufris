# Build psutil-backed host metrics collector (HostStats)

- PRIORITY: 30
- TAGS: feature, backlog, dashboard, monitoring
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Build a psutil-backed host metrics collector behind a `HostStats` pydantic model
and a fakeable collector seam, so the dashboard backend can serve a read-only
snapshot of host stats.

First-slice stats: hostname, CPU percent (overall + per-core), memory + swap,
disk usage per mount, load average, uptime/boot time, network IO counters,
static OS/kernel info. Temperatures and per-process tables are a follow-up.

## Steps

- [x] Add `psutil` (`uv add psutil`, `uv lock`); confirm it resolves in the dev
      shell.
- [x] Define pydantic v2 models in `scufris/metrics.py`: `MemStats`, `SwapStats`,
      `DiskUsage`, `NetIO`, and the top-level `HostStats` (fields per Goal:
      hostname, os/kernel, cpu_percent, per_cpu_percent, mem, swap, disks,
      load_avg, uptime_seconds, net, sampled_at).
- [x] Define the collector seam: a `Collector` protocol with `sample() ->
      HostStats`, and a `PsutilCollector` implementation that reads psutil. Pick
      the non-blocking CPU-percent strategy (prime once at construction, then
      `cpu_percent(interval=None)` deltas) so `sample()` never blocks.
- [x] Handle graceful degradation (a missing sensor / permission) without
      crashing the sample.
- [x] Tests in `tests/test_metrics.py`: a fake `Collector` asserting the
      `HostStats` shape/serialization, plus a light real-`PsutilCollector` smoke
      test that fields populate.
- [x] Run `ruff check .`, `mypy .`, `pytest` green.

## Definition of Done

- `PsutilCollector().sample()` returns a fully populated `HostStats` with the
  first-slice fields, sourced from psutil, without blocking.
- Collection sits behind the `Collector` protocol so tests fake the seam (not
  psutil) and the source is swappable.
- ruff, mypy and pytest are green.

## Notes

- Spike: tasks/20260719-153045/SPIKE.md (recommends psutil behind a
  `sample() -> HostStats` seam; pydantic v2 model; fake the collector in tests).
- Add `psutil` via `uv add psutil` then `uv lock`; re-enter `nix develop`.
  Verify it resolves inside the nix runtime closure (`nix run .#scufris`).
- Keep collection behind one interface (a `Collector` protocol or a single
  `sample()`), so the source is swappable and tests fake the seam, not psutil.
- Harness-first: a test that samples a faked collector and asserts the model
  shape; a light real-psutil smoke test.
- Decide CPU-percent interval strategy (blocking `cpu_percent(interval)` vs
  non-blocking deltas) during /plan.
- Depends on nothing; pairs with the dashboard backend task from
  spike tasks/20260719-153034 which serves this over HTTP.

## Implementation

- `scufris/metrics.py`: pydantic models (`MemStats`, `SwapStats`, `DiskUsage`,
  `NetIO`, `HostStats`), a `runtime_checkable` `Collector` protocol, and
  `PsutilCollector`. CPU percent is primed in `__init__` and read non-blocking
  (`interval=None`) in `sample()`, so sampling never blocks. Disk stat failures
  and a missing `getloadavg` degrade gracefully instead of failing the sample.
- Added `scufris/__init__.py` so the package resolves cleanly for mypy.
- Deps: `psutil` (runtime) + `types-psutil` (dev) via `uv add`; `uv.lock`
  updated, verified importable through the uv2nix-built venv (psutil 7.2.2).
- Tests: `tests/test_metrics.py` - a `FakeCollector` pinning the model
  shape/JSON round-trip (also reused by the backend task) and a real
  `PsutilCollector` smoke test. ruff + mypy + pytest green.
