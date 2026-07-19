# Fill the Load card + fixed-size Disks (dash when idle)

- STATUS: CLOSED
- PRIORITY: 13
- TAGS: feature, backlog, dashboard, ui

## Goal

Two small stats-page polish changes: (1) fill the Load-average card so it does
not feel empty, and (2) stop the Disks card (and cards generally) from
growing/shrinking as IO/temp subsections blink in and out - render stable rows
with a dash when a value is absent, and give cards a consistent min-height.

## Scope (user, 2026-07-19; Load-card contents confirmed via questionnaire)

- Load-average card: KEEP the 1/5/15 load numbers as the headline; ADD rows for:
  - process count ("tasks: N", `len(psutil.pids())` - cheap, NOT the full process
    table which is tatr 20260719-182901),
  - context-switches/sec + interrupts/sec (`psutil.cpu_stats()` deltas, same rate
    pattern as net/disk),
  - uptime (mirror `uptime_seconds` here; it is currently only in the header).
  - (User did NOT want load-vs-core bars - skip those.)
- Disks card stability: the card resizes when the `io`/`temp` subsections appear
  only on active polls. Make it static: always render the real disks (e.g.
  `nvme0n1`) with their usage/IO/temp; show `-` when a value is absent this poll
  instead of dropping the row/section. Probe the actual `disk_io` device names on
  this host and pick a stable filter (drop loop/ram/dm/sr noise).
- Give `.card` a min-height so cards do not collapse/jump as content varies.

## Steps

- [x] Backend (`scufris/metrics.py`): `HostStats` gains `process_count: int` and
      `cpu_activity` (`ctx_switches_per_sec`, `interrupts_per_sec`), default-empty.
      Collector: `process_count = len(psutil.pids())`; `cpu_activity` from
      `psutil.cpu_stats()` deltas over a persisted monotonic timestamp.
- [x] Frontend types (`common.ts`): add `process_count` + `cpu_activity`.
- [x] Load card: keep the 1/5/15 headline; add rows tasks (process_count),
      ctx/s, interrupts/s, uptime (`formatUptime`).
- [x] Disks card stability: render a STABLE set of base physical disks (drop
      `loop*`/`ram*`/`dm-*`/`sr*` and partitions - a device whose name has another
      device as a strict prefix; on this host that leaves `nvme0n1`). Always show
      each base disk's IO row (rate, or `-` when idle) instead of filtering by
      traffic; keep the temp section (stable). Add a `.card` min-height.
- [x] Tests: backend `cpu_activity` rate across two samples + `process_count > 0`;
      jsdom load-card rows (tasks/ctx/uptime) and disks IO rows always present
      with `-` when idle.
- [x] LIVE serve smoke; `ruff`/`mypy`/`pytest` + `npm run ci` green.

## Definition of Done

- The Load card shows load averages + tasks + ctx/s + interrupts/s + uptime.
- The Disks card shows a stable row set (base disks always, `-` when idle) and
  does not resize as IO blinks; cards have a consistent min-height.
- Serve-verified; names escaped; python + `npm run ci` green.

## Notes

- Backend: `scufris/metrics.py` gains `process_count: int` and a `cpu_activity`
  (ctx/interrupt per-sec) on `HostStats`, computed with persisted-counter deltas
  (reuse the monotonic-clock rate pattern). Default-empty for back-compat.
- Frontend: `common.ts` types + `stats-view.ts` loadCard/disksCard; keep names
  escaped and the module side-effect-free; update the jsdom tests.
- Harness-first: backend rate test across two samples; jsdom test for the new
  load rows and the dash-when-idle disk rows.
- Builds on the card rework (tatr 20260719-190533).

## Implementation

- Backend: `HostStats` gains `process_count` and a `cpu_activity` model
  (ctx_switches/interrupts per sec). The collector reads `len(psutil.pids())`
  and computes activity from `psutil.cpu_stats()` deltas over a persisted
  monotonic timestamp (first sample = 0). Both default-empty for back-compat.
- Load card: keeps the 1/5/15 headline and adds rows - tasks (process count),
  ctx switches/s, interrupts/s, uptime.
- Disks card: an `io` section that always renders the STABLE base disks (drop
  loop/ram/dm/sr noise and partitions via a strict-prefix rule; on this host
  that leaves `nvme0n1`), showing a `-` when a disk is idle rather than dropping
  the row/section - so the card no longer resizes as IO blinks. `.card` gains a
  `min-height` so cards do not collapse/jump.
- Tests: backend `cpu_activity` rate across two samples + `process_count > 0`;
  jsdom load-card rows and the base-disk/dash behavior (nvme0n1 shown idle as
  `-`, partitions + loop noise dropped). 11 jsdom tests; python + `npm run ci`
  green.

### Live verification (DoD)

On this host `/api/stats`: process_count 511, cpu_activity ctx ~20k/s +
interrupts ~6k/s, disk_io includes `nvme0n1` (frontend shows just it). `/stats/`
serves 200. Load card is full; Disks shows nvme0n1 stably.
