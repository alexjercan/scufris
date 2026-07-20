# Review: btop-style process view

## Round 1 - 20260719

Scope: `scufris/processes.py` (new), `scufris/app.py` (`/api/processes`),
`tests/test_processes.py`, `tests/test_app.py`, `web/src/processes-view.ts` (new)
+ its test, `web/src/common.ts` (shared formatBytes), `web/src/stats.ts` /
`stats.html` / `style.css`.

### Correctness

- Proven live: `/api/processes` returned 514 procs aggregated into 40 groups
  (`.claude-wrapped x8 @107%`, `firefox 720MB`, `python3.14 x4`), sorted by cpu,
  instances capped at 5. The stats page shows the `#processes` section below the
  cards, as the user asked.
- The aggregation logic is pure and well-tested: grouping happens BEFORE the
  top-N cut so a group's `count`/sums cover all its processes (a test asserts
  `total`/`count` stay whole while groups/instances are capped). Sorted by cpu
  then mem.
- Stateful cpu% is handled correctly and cheaply: `psutil.process_iter` caches
  Process objects so `cpu_percent` is a real delta, primed in the collector's
  constructor - no bespoke per-pid cache needed.
- The endpoint is decoupled from `/api/stats` and injected (`ProcessCollector`),
  so tests fake it; the heavy feed can poll independently.
- Frontend: collapsible + sortable; expand and sort state persist across the poll
  re-renders (module state, reset hook for tests). Names/usernames escaped; a
  jsdom test proves a hostile process name injects no element. `renderProcesses`
  is import-side-effect-free.
- Shared `formatBytes` moved to `common.ts` (used by both views) - removed the
  duplication rather than copy it. ruff/mypy/pytest + `npm run ci` (15 jsdom
  tests) green.

### Observations (non-blocking)

- MINOR: the group row uses `innerHTML` with escaped values (consistent with the
  rest of the codebase; the injection test covers it). A future refactor to build
  text nodes would remove the pattern entirely, but it is not required here.
- MINOR: two independent pollers now run on the stats page (`startStats` +
  `startProcesses`), each doing its own `loadConfig`. Fine for one user; a shared
  poll clock is a possible tidy-up.
- NIT: `top_groups=40` / `top_instances=5` are sensible defaults; payload was
  small in practice. Tunable if needed.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: `/api/processes` aggregates by application
(grouping before capping), and the stats page shows a live, collapsible, sortable
grouped process view below the cards with escaped names; aggregation, endpoint and
table are tested and it is serve-verified on this host. MINOR items are optional
tidy-ups.
