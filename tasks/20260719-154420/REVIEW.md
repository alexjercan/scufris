# Review: Build psutil-backed host metrics collector

## Round 1 - 20260719

Scope: `scufris/metrics.py`, `scufris/__init__.py`, `tests/test_metrics.py`,
`pyproject.toml` / `uv.lock` (added psutil + types-psutil).

### Correctness

- CPU percent is primed for BOTH the total and per-cpu counters in `__init__`
  (psutil keeps separate internal last-times for each), so the first `sample()`
  returns a real non-blocking delta for both. Correct.
- `sample()` performs no blocking calls (`interval=None` throughout); uptime is
  clamped non-negative; disk stat errors and a missing `getloadavg` degrade to
  skip/zeros instead of failing the snapshot. Matches the Definition of Done.
- `HostStats` round-trips to JSON (`model_dump(mode="json")`), pinned by the
  fake-collector test - important since the backend serves exactly this.

### Observations (non-blocking)

- LOW: `psutil.net_io_counters()` can theoretically return `None` on a host with
  no NIC counters; we assume a namedtuple. On any real host loopback exists, so
  this will not fire for the target use, but a defensive guard would be tidier.
  Left as a follow-up rather than blocking the first slice.
- LOW: the smoke test's `mem.used >= 0` assertion is trivially true; the
  surrounding assertions (total > 0, cpu in range, tz-aware timestamp) carry the
  real signal, so coverage is adequate.
- NOTE: first sample right after construction has a tiny delta window so CPU can
  read ~0; documented in the module and harmless given the frontend polls.

### Verdict

APPROVE. The collector meets its Definition of Done, sits behind a fakeable
seam, and the checks (ruff, ruff format, mypy, pytest) are green. The two LOW
observations are optional hardening, appropriate as future tasks if they ever
matter.
