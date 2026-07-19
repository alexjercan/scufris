# Sparkline history: btop-style mini-graph on each stats card

- STATUS: OPEN
- PRIORITY: 5
- TAGS: feature,backlog,dashboard

## Goal

Add a btop-style live mini-graph (sparkline) to each stats card, so cpu / load /
gpu / memory / disk-io / network trend over the recent window is visible at a
glance alongside the current numbers.

## Approach (re-cut from the deferred backend-sampler plan)

The original note called for a backend `BackgroundSampler` + ring buffer +
`GET /api/history`. That buys persistence across reloads and shared history
across clients - a benefit nobody asked for - at the cost of a background task,
FastAPI lifespan wiring, memory bounds and a new endpoint. "A minigraph like
btop" is a live rolling window that fills from start, not persisted history, so
the honest, simpler fit is a CLIENT-SIDE ring buffer: the page already polls
`/api/stats` every `poll_seconds`, so accumulate each poll's value into a bounded
per-series array and draw a sparkline from it. Zero backend change; history is
since-page-load, exactly like btop since-start. If cross-reload persistence is
ever wanted, the backend sampler remains a clean later follow-up.

## Steps

- [ ] `stats-view.ts`: client-side history - a module `_history: Map<string,
      number[]>`, `HISTORY_LEN`, a `push(key, value)` helper (append + cap), and
      a `_resetStatsHistory()` export (test-reset hook, per
      `persistent-ui-state-needs-a-test-reset-hook`).
- [ ] `stats-view.ts`: a pure `sparkline(values, max?, sevClass?)` returning an
      inline `<svg class="spark">` (filled area polygon + polyline, viewBox
      100x30, `preserveAspectRatio=none` so CSS scales it). Percent series pass
      `max=100`; rate/load series autoscale to the window max (min 1). Empty-safe.
- [ ] Wire one sparkline into each main card, pushing its series in `renderCards`
      then passing the slice to the card builder: CPU (cpu_percent, max 100),
      Load average (load_avg[0], autoscale), GPU (util_percent, max 100; keyed
      per gpu index), Memory (mem.percent, max 100), Disks (summed base-disk
      read+write bytes/s, autoscale), Network (summed sent+recv bytes/s,
      autoscale). Latest-value severity colours the line where a percent applies.
- [ ] `style.css`: theme `.spark` / `.spark__area` / `.spark__line` (+ severity
      variants), sized to sit under the card value/bar without changing card
      height (respect the fixed-size-cards lesson).
- [ ] `stats-view.test.ts` (jsdom): `sparkline` (point count = values length,
      area + line present, empty-safe, `max` clamps y), and that two successive
      `renderCards` polls grow the CPU sparkline's point count; `_resetStatsHistory`
      in `beforeEach`.
- [ ] LIVE serve smoke: each card shows a mini-graph that fills across polls;
      `npm run ci` + `ruff`/`mypy`/`pytest` green.

## Definition of Done

- Each stats card (CPU, Load, GPU, Memory, Disks, Network) shows a btop-style
  mini-graph that fills as polls accumulate: percent cards fixed to 0-100,
  rate/load cards autoscaled to their window. Card height is unchanged; host
  strings stay escaped; jsdom tests + `npm run ci` + python checks all green;
  serve-verified on this host.

## Notes

- Spike: tasks/20260719-180507/SPIKE.md (chose "current-first, history later";
  this is that follow-up, re-cut to client-side per the Approach above).
- No backend change. `/api/stats` already carries everything graphed.
- Frontend render stays side-effect-free for jsdom
  (`side-effect-free-module-for-jsdom-tests`); keep host strings escaped.
