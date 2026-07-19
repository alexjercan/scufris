# Richer host metrics: GPU, sensors, per-core/net/disk detail

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,backlog,dashboard,monitoring

## Goal

Extend the host metrics collector and `HostStats` with GPU, temperature/fan
sensors, per-core frequency, per-NIC network rates, and per-disk IO rates, and
surface them as new cards on the stats page. All the "cheap" richer metrics that
ride the existing summary poll (the heavy process table is a separate task).

## Notes

- Spike: tasks/20260719-180507/SPIKE.md (RECOMMENDED direction; user decisions
  captured there).
- GPU: shell out to `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
  (util, VRAM used/total, temp, power, clocks) -> a `GpuStats` list; resolve via
  `shutil.which`, empty list when absent. NOT NVML/pynvml (driver-linkage
  friction on NixOS - see spike). Per-process GPU is deferred.
- Sensors: `psutil.sensors_temperatures()` (works here: coretemp per-core +
  package, nvme, acpitz), plus `sensors_fans()`/`sensors_battery()` where
  present; render only what exists.
- Extended: per-core freq (`cpu_freq(percpu=True)`), per-NIC net RATES (delta of
  `net_io_counters(pernic=True)` over time), per-disk IO RATES
  (`disk_io_counters(perdisk=True)`). Rates use the collector's persisted
  previous counters + timestamp (the collector is already stateful for cpu%).
- Keep it on `GET /api/stats` (light summary); the process table is task
  20260719-182901. Stats page gains GPU / sensors / freq / per-NIC /
  per-disk cards, matching the existing theme.
- Harness-first: fake the collector seam; test the nvidia-smi parse against a
  captured CSV sample; test rate deltas across two samples.
- Reuses the `Collector`/`HostStats` seam (tatr 20260719-154420).
