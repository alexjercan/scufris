# btop-style process view: grouped-by-application, collapsible

- STATUS: OPEN
- PRIORITY: 15
- TAGS: feature,backlog,dashboard,ui

## Goal

Add a btop-style process view: a stateful `GET /api/processes` that aggregates
processes BY APPLICATION, and a collapsible, sortable grouped-by-application
process table on the stats page.

## Placement (user, 2026-07-19)

- The process view lives on the STATS page, BELOW the cards grid (its own
  section, not a card). Own `GET /api/processes` endpoint, polled independently.

## Steps

- [ ] Backend models (`scufris/processes.py`): `ProcessInstance` (pid, username,
      cpu_percent, mem_rss, num_threads, status), `ProcessGroup` (name, count,
      cpu_percent, mem_rss, instances), `ProcessList` (groups, total). A pure
      `aggregate_processes(rows) -> ProcessList` (group by name, sum cpu/mem,
      count, top-K instances per group by cpu, top-N groups) - unit-testable.
      A `ProcessCollector` protocol + `PsutilProcessCollector` that gathers rows
      via `psutil.process_iter(['pid','name','username','cpu_percent',
      'memory_info','num_threads','status'])` (primed in `__init__` so cpu% is a
      real delta) and calls `aggregate_processes`.
- [ ] Backend endpoint: `GET /api/processes` in `create_app`, driven by an
      injected `ProcessCollector` (default `PsutilProcessCollector`).
- [ ] Frontend `web/src/processes-view.ts` (side-effect-free): types mirroring
      the payload; `renderProcesses(list)` builds a collapsible, sortable
      grouped-by-application table; `startProcesses()` polls `/api/processes`.
      Persist expanded-group + sort state across re-renders (module state).
      Escape name/username.
- [ ] stats.html: add a `<section id="processes">` below `<main id="cards">`;
      `stats.ts` entry also calls `startProcesses()`; theme it in `style.css`.
- [ ] Tests: backend `aggregate_processes` (grouping, sums, top-N, instance cap)
      with fake rows + `/api/processes` via a fake collector; jsdom
      `renderProcesses` (group rows, expand reveals instances, sort reorders,
      hostile name/user injects nothing).
- [ ] LIVE serve smoke: `/api/processes` returns real grouped data; the stats
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
