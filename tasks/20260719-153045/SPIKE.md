# Spike: Host metrics collection approach

- DATE: 20260719-153045
- STATUS: RECOMMENDED
- TAGS: spike, backlog, dashboard, monitoring

## Question

How should Scufris collect host metrics on a Linux/NixOS machine - which stats
matter, and via which source (a Python library like psutil, reading /proc + /sys
directly, or shelling out to system tools) - and what internal shape should a
metric sample take so both the dashboard and the agent can consume it?

## Context

The monitoring dashboard is the first Scufris pillar and the immediate goal is a
simple dashboard that just runs and shows host stats (user, 2026-07-19). The
dashboard is read-only: the backend samples metrics and serves them over a GET
endpoint; there is no user-triggered "RUN". The target host is NixOS, so any
dependency must package through the `uv2nix` flake. The same samples later feed
the agent as a read-only tool it can query.

## Options considered

- **psutil** - one well-maintained library covering CPU, memory, swap, disk
  usage + IO, network IO, load average, per-process info, sensors/temperatures,
  boot time. Cross-platform, pure Python API over a small C extension. Ships
  manylinux wheels, so `uv2nix` with `sourcePreference = "wheel"` (already set in
  `flake.nix`) pulls it without a source build; nixpkgs also has
  `python3Packages.psutil` as a fallback. Pros: one dependency, broad coverage,
  trivially faked in tests (patch the collector, not psutil internals). Cons: a
  compiled dep (low risk given wheels); a couple of gaps (some sensors) may still
  need a direct read.
- **Direct /proc and /sys reads** - zero dependencies, full control. Read
  `/proc/stat`, `/proc/meminfo`, `/proc/loadavg`, `/sys/class/thermal`, etc.
  Pros: no deps, nothing to package. Cons: a meaningful amount of Linux-specific
  parsing code to write and maintain for what psutil already solves; CPU percent
  needs manual delta sampling between reads.
- **Shell out to system tools** - parse `free`, `df`, `sensors`, `ip -s`, etc.
  Pros: reuses familiar tools. Cons: every tool must be in the nix runtime
  closure, output formats are brittle to parse and version-dependent, and it
  spawns subprocesses on every sample. Worst fit for a read-often dashboard.
- **Do nothing yet** - ship a static/fake stats payload first. Costs nothing but
  gives no real signal; only useful as a throwaway to unblock the frontend.

## Recommendation

Use **psutil**, wrapped behind a small collector interface that returns a
pydantic `HostStats` model. psutil gives near-complete coverage from one
dependency that packages cleanly through the existing wheel-preferring uv2nix
setup, and the C-extension risk is negligible because manylinux wheels avoid a
source build. Keep collection behind one seam (a `Collector` protocol / single
`sample() -> HostStats` function) so: (1) the source can be swapped or a /proc
read dropped in for a gap without touching the dashboard, and (2) tests fake the
collector, staying harness-first per AGENTS.md. Where psutil lacks a datum
(certain temperature sensors), fill that one field with a direct `/sys` read
behind the same interface rather than switching approaches.

First-slice stat set (keep it small, expand later): CPU percent (overall, maybe
per-core), memory + swap (used/total/percent), disk usage per mount, load
average, uptime/boot time, network IO counters. Hostname/OS/kernel as static
info. Temperatures and per-process tables are a follow-up, not the first slice.

Proposed shape (pydantic v2, already a dependency):

```python
class HostStats(BaseModel):
    hostname: str
    cpu_percent: float
    per_cpu_percent: list[float]
    mem: MemStats            # total, used, available, percent
    swap: SwapStats          # total, used, percent
    disks: list[DiskUsage]   # mountpoint, total, used, percent
    load_avg: tuple[float, float, float]
    uptime_seconds: float
    net: NetIO               # bytes_sent, bytes_recv (counters)
    sampled_at: datetime
```

## Open questions

- Sampling model: per-request sampling (simplest; sample on each GET) vs a
  background sampler with a cached latest value (smoother, needed once the
  frontend polls fast). Start with per-request; revisit if the poll interval
  gets tight. This is the single reversible lever.
- Does psutil resolve every wanted datum inside the nix runtime closure (sensors
  especially)? Verify at implementation with `nix run`; fall back to a `/sys`
  read for any gap.
- CPU percent needs an interval to be meaningful (psutil `cpu_percent(interval)`
  blocks, or non-blocking deltas between samples). Decide during implementation.

## Next steps

Direction-level task this spike seeded, for `/plan` to break into steps:

- tatr 20260719-154420: build the psutil-backed host metrics collector behind a
  `HostStats` pydantic model and a fakeable collector seam.

(The dashboard/backend that serves these stats over HTTP is seeded by the
dashboard-style spike [[20260719-153034]]; this spike owns only the collector.)
