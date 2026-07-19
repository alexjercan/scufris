# Richer host metrics: GPU, sensors, per-core/net/disk detail

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,backlog,dashboard,monitoring

## Goal

Extend the host metrics collector and `HostStats` with GPU, temperature/fan
sensors, per-core frequency, per-NIC network rates, and per-disk IO rates, and
surface them as new cards on the stats page. All the "cheap" richer metrics that
ride the existing summary poll (the heavy process table is a separate task).

## Steps

- [ ] Models (`scufris/metrics.py`): add `GpuStats` (name, util %, mem used/total
      + %, temp, power + limit, sm/mem clocks), `SensorReading` (label, current,
      high?, critical?) grouped per chip, `FanReading`, `NetIfRate` (name, sent/s,
      recv/s), `DiskIoRate` (name, read/s, write/s). Extend `HostStats` with
      `gpus`, `temps`, `fans`, `per_cpu_freq_mhz`, `net_interfaces`, `disk_io` -
      all DEFAULTING to empty so existing fixtures/constructions still work.
- [ ] `PsutilCollector`: GPU via a `nvidia-smi --query-gpu=... --format=csv,
      noheader,nounits` subprocess behind an injectable runner seam (resolve with
      `shutil.which`, timeout, empty list on absence/parse error); temps/fans via
      `psutil.sensors_temperatures/_fans`; per-core freq via `cpu_freq(percpu)`;
      per-NIC + per-disk RATES from persisted previous counters + a monotonic
      timestamp (first sample -> 0 rates). Keep it non-blocking.
- [ ] Backend tests: nvidia-smi CSV parse via a FAKE runner (canned sample);
      rate deltas across two samples (injected/fake counters); temps mapping;
      `/api/stats` still serializes. Extend one fixture to include the new fields.
- [ ] Frontend types + cards (`web/src/common.ts` + `stats-view.ts`): extend the
      `HostStats` TS type; add GPU, Sensors, CPU-frequency, Network-interfaces and
      Disk-IO cards; format byte rates; ESCAPE host-derived names (gpu / nic /
      disk / sensor labels) per the escaping lesson.
- [ ] Frontend tests (jsdom): the new cards render from a fixture; a hostile GPU
      or interface name injects no element.
- [ ] LIVE VERIFY on this host: `/api/stats` includes the RTX 3060 Ti gpu, real
      temps (coretemp/nvme), per-core freq, and non-null net/disk rates on the
      second poll; the stats page renders the new cards. Record evidence.
- [ ] `ruff`/`mypy`/`pytest` + `npm run ci` green; serve smoke.

## Definition of Done

- `/api/stats` returns `gpus`, `temps`, `fans`, `per_cpu_freq_mhz`,
  `net_interfaces`, `disk_io` populated from real host data (GPU via nvidia-smi,
  degrading to empty when absent; rates correct across polls).
- The stats page shows GPU, Sensors, CPU-frequency, Network and Disk-IO cards
  with live data (serve-verified on this host); host-derived names are escaped.
- ruff, mypy, pytest and `npm run ci` (incl. jsdom tests) are green.

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
