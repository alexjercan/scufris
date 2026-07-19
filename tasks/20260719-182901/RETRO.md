# Retro: btop-style process view

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The spike had already fixed the shape (grouped-by-application, expandable,
  sortable), so this was a clean build. Splitting the pure `aggregate_processes`
  from the psutil gathering meant the grouping/top-N logic was fully unit-tested
  without a real process list.
- `psutil.process_iter`'s internal Process cache gave stateful cpu% deltas for
  free (primed once in the constructor) - no bespoke per-pid cache, which the
  task had budgeted for.
- The earlier side-effect-free-module discipline paid off a third time: a new
  `processes-view.ts` with exported `renderProcesses` was jsdom-testable, and the
  collapsible/sortable behaviour (expand-on-click reveals instances, MEM sort
  reorders) was asserted directly.
- Moving the duplicated `formatBytes` to `common.ts` while touching both views
  removed the smell rather than copying it.

## What went wrong / friction

- The module-level expand/sort state (needed so a poll re-render does not collapse
  what the user opened) leaks between jsdom test cases - added a small
  `_resetProcessState()` test hook and reset it in `beforeEach`. A clean, honest
  way to test persistent UI state.

## Lessons

- `psutil-process-iter-caches-cpu-percent`: `psutil.process_iter` reuses Process
  objects internally, so `cpu_percent` is a real delta across calls with no
  per-pid cache of your own - just prime it once (iterate at startup) and read on
  each sample.
- `persistent-ui-state-needs-a-test-reset-hook`: module-level UI state (expanded
  set, sort key) that must survive re-renders leaks across jsdom tests; export a
  small reset and call it in `beforeEach`.

## Follow-ups

- Optional tidy-ups from REVIEW: a shared poll clock instead of two independent
  pollers on the stats page; build text nodes instead of escaped innerHTML.
- Remaining backlog: sparkline history (182915), agent chat-page spike (180528).
