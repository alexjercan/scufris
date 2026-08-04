# btop-style process view: grouped-by-application, collapsible

- PRIORITY: 15
- TAGS: feature, backlog, dashboard, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Add a btop-style process view: a stateful `GET /api/processes` that aggregates
processes BY APPLICATION, and a collapsible, sortable grouped-by-application
process table on the stats page.

## Placement (user, 2026-07-19)

- The process view lives on the STATS page, BELOW the cards grid (its own
  section, not a card). Own `GET /api/processes` endpoint, polled independently.

## Steps

- [x] Backend models (`scufris/processes.py`): `ProcessInstance` (pid, username,
      cpu_percent, mem_rss, num_threads, status), `ProcessGroup` (name, count,
      cpu_percent, mem_rss, instances), `ProcessList` (groups, total). A pure
      `aggregate_processes(rows) -> ProcessList` (group by name, sum cpu/mem,
      count, top-K instances per group by cpu, top-N groups) - unit-testable.
      A `ProcessCollector` protocol + `PsutilProcessCollector` that gathers rows
      via `psutil.process_iter(['pid','name','username','cpu_percent',
      'memory_info','num_threads','status'])` (primed in `__init__` so cpu% is a
      real delta) and calls `aggregate_processes`.
- [x] Backend endpoint: `GET /api/processes` in `create_app`, driven by an
      injected `ProcessCollector` (default `PsutilProcessCollector`).
- [x] Frontend `web/src/processes-view.ts` (side-effect-free): types mirroring
      the payload; `renderProcesses(list)` builds a collapsible, sortable
      grouped-by-application table; `startProcesses()` polls `/api/processes`.
      Persist expanded-group + sort state across re-renders (module state).
      Escape name/username.
- [x] stats.html: add a `<section id="processes">` below `<main id="cards">`;
      `stats.ts` entry also calls `startProcesses()`; theme it in `style.css`.
- [x] Tests: backend `aggregate_processes` (grouping, sums, top-N, instance cap)
      with fake rows + `/api/processes` via a fake collector; jsdom
      `renderProcesses` (group rows, expand reveals instances, sort reorders,
      hostile name/user injects nothing).
- [x] LIVE serve smoke: `/api/processes` returns real grouped data; the stats
      page shows the process view below the cards. `ruff`/`mypy`/`pytest` +
      `npm run ci` green.

## Definition of Done

- `GET /api/processes` returns processes aggregated by application (count, summed
  cpu/mem, top-K instances, top-N groups; grouping before capping).
- The stats page shows, below the cards, a collapsible + sortable
  grouped-by-application process view with live data; names/users escaped.
- Tests green (aggregation + endpoint + jsdom table); serve-verified on this host.

## Notes

- Spike: tasks/20260719-180507/SPIKE.md (user chose grouped-by-application:
  merge same-name processes into one row with a count + summed cpu/mem,
  expandable to instances; sortable; "more intuitive than htop").
- Backend: iterate `process_iter` (stateful per-process cpu% deltas across
  polls - keep a process cache), group by name into `{name, count, cpu_percent
  (sum), mem (sum), instances:[{pid,user,cpu,mem,threads,status}...]}`, return
  the top-N groups by resource use (grouping BEFORE capping so group totals are
  correct). Its own endpoint so the heavy feed is decoupled from the light
  `/api/stats` and can poll at a slower cadence.
- Frontend: a collapsible table on the stats page - each app row shows name,
  summed cpu%, summed mem, instance count, with an expand toggle revealing its
  instances; sortable by cpu/mem; escape process names/users (host-derived
  strings) like the other cards. Fits the scufris theme.
- Harness-first: fake the collector; assert grouping/aggregation and top-N;
  jsdom test the collapsible table (expand/collapse, sort, escaping).
- Tune top-N and poll cadence during /plan; measure payload. Depends on the
  richer-metrics task only loosely (can proceed in parallel).

## Implementation

- `scufris/processes.py`: `ProcessInstance` / `ProcessGroup` / `ProcessList`
  models, a pure `aggregate_processes(rows) -> ProcessList` (group by name, sum
  cpu/mem, count, top-K instances per group, top-N groups; grouping BEFORE
  capping), a `ProcessCollector` protocol, and `PsutilProcessCollector`
  (process_iter with the fields; cpu% primed in __init__ so the first sample is a
  real delta).
- Backend: `GET /api/processes` in `create_app`, driven by an injected
  `ProcessCollector` (default psutil), decoupled from the light `/api/stats`.
- Frontend: `web/src/processes-view.ts` (side-effect-free) - a collapsible,
  sortable grouped-by-application table; expand/sort state persists across the
  poll re-renders; escapes name/username. `startProcesses()` polls
  `/api/processes`. Added `<section id="processes">` below the cards on
  stats.html; stats.ts starts it. Shared `formatBytes` moved to `common.ts`.
- Tests: backend `aggregate_processes` (grouping/sums/top-N/instance cap) +
  `/api/processes` via a fake collector; jsdom `renderProcesses` (group rows,
  expand-reveals-instances on click, sort reorders, hostile-name injection
  guard). 15 jsdom tests total; python + `npm run ci` green.

### Live verification (DoD)

On this host `/api/processes`: 514 procs -> 40 groups, correctly aggregated
(`.claude-wrapped x8 @107%`, `firefox 720MB`, `python3.14 x4`), sorted by cpu,
instances capped at 5. The stats page carries the `#processes` section below the
cards.
