# Build psutil-backed host metrics collector (HostStats)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,backlog,dashboard,monitoring

## Goal

Build a psutil-backed host metrics collector behind a `HostStats` pydantic model
and a fakeable collector seam, so the dashboard backend can serve a read-only
snapshot of host stats.

First-slice stats: hostname, CPU percent (overall + per-core), memory + swap,
disk usage per mount, load average, uptime/boot time, network IO counters,
static OS/kernel info. Temperatures and per-process tables are a follow-up.

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
