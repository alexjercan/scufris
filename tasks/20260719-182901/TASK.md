# btop-style process view: grouped-by-application, collapsible

- STATUS: OPEN
- PRIORITY: 15
- TAGS: feature,backlog,dashboard,ui

## Goal

Add a btop-style process view: a stateful `GET /api/processes` that aggregates
processes BY APPLICATION, and a collapsible, sortable grouped-by-application
process table on the stats page.

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
