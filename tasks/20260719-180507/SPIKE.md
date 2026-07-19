# Spike: richer live system stats (GPU, sensors, processes, btop-style)

- DATE: 20260719-180507
- STATUS: RECOMMENDED
- TAGS: spike, backlog, dashboard, monitoring

## Question

How do we grow the stats view from the current basic snapshot into a rich,
intuitive, live system monitor - GPU (nvtop-style), temperature sensors, a
collapsible btop-style process view, plus deeper CPU/mem/net/disk detail - and
what do we collect, from where, and how do we present it?

## Context

Today `scufris/metrics.py` is a stateful `PsutilCollector.sample() -> HostStats`
(CPU + per-core %, mem, swap, disks, load, net totals, uptime), served at
`GET /api/stats`; the stats page (now its own page at `/stats/`, task
20260719-180543) polls it and renders cards. This spike covers collection,
data/sampling shape, and the btop-inspired UI.

**Host probe (this machine, 2026-07-19)** grounded the options:

- GPU: NVIDIA GeForce RTX 3060 Ti (8 GB). `nvidia-smi` works on PATH; Python NVML
  libs (`pynvml`/`nvitop`/`gpustat`) are NOT installed. No AMD.
- Temps: `psutil.sensors_temperatures()` already returns coretemp (per-core +
  package), nvme (Composite), acpitz, asus - NO extra binary. `sensors_fans()`
  returns nothing here; `sensors_battery()` is None (desktop).
- Processes: `psutil.process_iter` yields ~525 procs with cpu%/rss/threads/user/
  name cleanly; per-core count 24.

**User decisions (questionnaire, 2026-07-19):**

- V1 scope: ALL four categories - process table, GPU, temps/sensors, and deeper
  CPU/mem/net/disk-IO. (Everything discoverable, per the original ask.)
- Process view: **grouped by application** - merge same-name processes into one
  row with a count and summed cpu/mem, expandable to the individual instances;
  sortable. This is the "more intuitive than htop" shape.
- History: **current-first, history later** - ship live current values now,
  design the seam for sparkline history, build the background sampler as a
  follow-up.
- GPU: **summary via `nvidia-smi`** (util, VRAM, temp, power, clocks) - not
  per-process GPU / NVML for now.

## Options considered

### GPU collection

- **`nvidia-smi` subprocess (RECOMMENDED, matches the user's pick).** Shell out
  to `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,
  temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits`,
  parse to a `GpuStats` list. Pros: works on this host right now, robust on NixOS
  (the driver wrapper is on PATH - same rationale as driving `codex` via its CLI,
  lesson `codex-exec-is-the-nixos-path`), no linkage risk, no new Python dep.
  Cons: a subprocess per poll; no per-process GPU usage (deferred); CSV parsing.
- **NVML via `pynvml`/`nvitop` (rejected for now).** Richer (per-process GPU/VRAM
  like nvtop, events), a clean Python API. Cons: needs `libnvidia-ml.so` loadable
  by the nix Python - driver-linkage friction on NixOS (nix-ld/LD_LIBRARY_PATH),
  the exact fragility the `nvidia-smi` path avoids. Revisit if per-process GPU is
  wanted later.
- **Do nothing** - loses a headline feature the user explicitly asked for.

### Temperatures / sensors

- **`psutil.sensors_temperatures()` (RECOMMENDED).** Already returns the real
  sensors here (coretemp per-core + package, nvme, acpitz) with no extra binary;
  include `sensors_fans()`/`sensors_battery()` for hosts that have them. GPU temp
  comes from the `nvidia-smi` query above. Cons: fan data absent on this host
  (would need lm-sensors config) - acceptable.
- **lm-sensors binary / raw `/sys/class/hwmon` reads (rejected).** No `sensors`
  binary installed; raw `/sys` reads reimplement what psutil already gives.

### Process collection + aggregation

- **Backend aggregates by application name, sends top-N groups + instances
  (RECOMMENDED).** The collector iterates processes (stateful, so per-process
  cpu% deltas are correct across polls), groups by name into `{name, count,
  cpu_percent (sum), mem (sum), instances: [top pids...]}`, and returns the top-N
  groups by resource use. Pros: correct group totals (grouping before capping),
  bounded payload, matches the chosen grouped-by-application UI directly. Cons:
  the grouping policy lives server-side.
- **Send all ~525 raw processes, group in the browser (rejected as primary).**
  Simpler backend, but a larger payload every poll and group totals depend on
  sending everything; keep as a fallback if client-side flexibility is wanted.
- **Flat top-N only (rejected).** Loses the grouping the user chose.

### Sampling model

- **Per-request, stateful collector (RECOMMENDED now).** The app holds one
  collector instance, so it already primes cpu% and can hold previous net/disk
  counters + timestamps to compute rates, and previous per-process cpu times.
  Keeps v1 simple (no background loop) while being correct for rates. The seam
  for history: keep all sampling in the collector so a `BackgroundSampler` can
  later call it on a timer into a ring buffer.
- **Background sampler + ring buffer now (deferred).** Needed for sparkline
  history (btop graphs) - the user chose to defer this. Build as a follow-up.
- **Per-request stateless (rejected).** Breaks rate/percent deltas.

### API payload shape

- **Keep `/api/stats` light; add `/api/processes` (RECOMMENDED).** Extend
  `HostStats` with the cheap additions (gpus, temps, per-core freq, per-nic net
  rates, per-disk IO rates) so the summary cards stay one request, and put the
  heavier process table behind its own `GET /api/processes` the stats page polls
  separately (possibly at a slower cadence). Cons: two endpoints.
- **One big `/api/stats` with everything (rejected).** Couples the heavy process
  list to the light cards and to a single poll cadence.

## Recommendation

Extend the existing stateful `Collector` seam, not replace it. Concretely:

1. **Richer summary metrics** in `HostStats` (served by the existing
   `/api/stats`): a `gpus: list[GpuStats]` from a `nvidia-smi` subprocess
   (util, VRAM used/total, temp, power, clocks; empty when absent), a
   `temps`/`sensors` block from `psutil.sensors_temperatures/_fans/_battery`,
   `per_cpu_freq`, per-NIC network **rates** (delta of `net_io_counters(pernic=
   True)` over time), and per-disk IO **rates** (`disk_io_counters(perdisk=
   True)`). All read once per sample; rates use the collector's persisted
   previous counters + timestamp. On the stats page these become new cards (GPU,
   Sensors, per-core freq, per-NIC, per-disk IO) alongside the existing ones.

2. **Process table** behind a new `GET /api/processes`: the collector iterates
   `process_iter`, computes per-process cpu% (stateful), and **aggregates by
   application name** into groups (count, summed cpu/mem, top instances), returns
   the top-N groups. The stats page renders a **collapsible grouped-by-application
   table**: each app is a row (name, summed cpu%, summed mem, instance count) with
   an expand toggle revealing its instances (pid, user, cpu%, mem, threads),
   sortable by cpu/mem. This is the btop-improved core.

3. **History is a designed seam, not built now**: keep sampling inside the
   collector so a later `BackgroundSampler` can poll it on a timer into a ring
   buffer and expose recent history for sparklines on the cards. (Follow-up.)

Why this beats the runners-up: it reuses the proven `Collector`/`HostStats`
architecture and the multi-page stats page; the `nvidia-smi` and psutil paths are
already working on this host with zero new fragile dependencies (per the
codex-CLI-on-NixOS precedent); server-side grouping gives correct totals for the
chosen UI; and splitting the heavy process feed off the light summary keeps the
dashboard responsive. It delivers all four categories the user wants while the
one genuinely heavy/architectural piece (history) is deferred behind a seam.

## Open questions

- **`nvidia-smi` availability in the packaged app / dev shell.** It is on the
  user's PATH (found via `shutil.which`, degrades gracefully if absent), but is
  NOT trivially added to the flake (it ships with the NVIDIA driver, not a
  nixpkgs package). Fine for `nix develop`/run-as-user; note for any packaged
  distribution. Resolve at implementation.
- **Process payload size / cadence.** Top-N group count and the `/api/processes`
  poll interval (likely slower than the 2s summary poll) - tune during `/plan`;
  measure the payload.
- **Per-process CPU% accuracy under grouping.** `process_iter` + cpu% deltas need
  a stable process cache between polls; decide the cache/ttl in implementation.
- **Fans/other sensors** are host-dependent (none exposed here) - render only
  what exists.
- **Per-process GPU usage** (nvtop's headline) is deferred with the NVML path; a
  future spike/task if wanted.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260719-182846: Richer host metrics - extend the collector + `HostStats`
  with GPU (`nvidia-smi`), temperatures/sensors, per-core freq, per-NIC net
  rates, and per-disk IO rates (stateful deltas); surface in `/api/stats`; add
  the GPU / sensors / extended cards to the stats page.
- tatr 20260719-182901: btop-style process view - a stateful `GET /api/processes`
  that aggregates processes by application (count, summed cpu/mem, instances,
  top-N) + a collapsible grouped-by-application, sortable process table.
- tatr 20260719-182915: (Deferred) sparkline history - a background sampler +
  ring buffer feeding mini history graphs on the cards. A follow-up (user chose
  current-first), after the richer-metrics task.

## Fix record

(Appended by each implementing task as it lands.)
