# Spike: richer live system stats (GPU, sensors, processes, btop-style)

- PRIORITY: 0
- TAGS: spike, backlog, dashboard, monitoring
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Question

How do we grow the stats view from the current basic snapshot into a rich,
intuitive, live system monitor - GPUs (usage / memory / temp / clocks, like
nvtop), temperature and fan sensors, and a processes view in the spirit of
htop/btop but collapsible and nicely packed - plus anything else discoverable
about the machine? What to collect, from where, and how to present it so it is
more intuitive than htop (think btop, improved).

## Context

Today: `scufris/metrics.py` is a psutil-backed `HostStats` (CPU + per-core, mem,
swap, disks, load, net, uptime) served over `GET /api/stats`; the dashboard is a
single-page webpack/TS/Tailwind SPA that polls and renders stat cards. This is a
BIG spike covering three axes at once:

- Collection: GPUs (NVIDIA via NVML/`nvidia-smi`, and think about AMD/Intel),
  temperature/fan sensors (lm-sensors / `psutil.sensors_temperatures` /
  `/sys/class/hwmon`), a full process table (psutil `process_iter`: pid, user,
  cpu%, mem, cmd, state, threads), and anything else useful (per-core freq,
  battery, GPU/VRAM, network per-nic, io). All must package in the nix runtime
  closure on NixOS (extra binaries or Python libs).
- Data shape / performance: polling many metrics (esp. the process table) is
  heavier than the current snapshot - decide payload shape, sampling cadence,
  and whether a background sampler + diff/delta is needed instead of
  per-request.
- Presentation: a btop-inspired UI - collapsible sections, a process view that
  groups/collapses (by app, by tree, by user) and is sortable, sparkline/gauge
  history, packed but readable. "More intuitive than htop" is the bar.

## What a good answer looks like

A recommended collection approach PER category (GPU, sensors, processes, misc)
with the NixOS packaging path for each; a data-model + sampling recommendation;
and a UI design direction (layout, collapsible process grouping, history
gauges). Concrete enough to seed several implementation tasks. Honest about what
is not worth doing or is host-dependent (e.g. no NVIDIA GPU present).

## Candidate directions to explore (diverge before converging)

- Extend the existing psutil collector vs add per-domain collectors behind the
  same seam; NVML/`nvidia-smi` for GPU; lm-sensors vs `/sys` reads for temps.
- Process view: flat sortable table vs collapsible tree/groups; how much history
  to keep client- vs server-side.
- UI: build our own btop-style widgets vs a charting lib; how it fits the
  current Tailwind theme.

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed implementation
  tasks, close the spike. This is intentionally ONE big spike (user's ask).
- Relates to the multi-page restructure (the richer stats likely live on their
  own page) and reuses the `Collector` seam from tatr 20260719-154420.
- User ask (2026-07-19): GPUs like nvtop, sensors, processes like htop but
  collapsible/nicely packed and more intuitive - btop, improved.
