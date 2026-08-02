# Build a filterable flow task board for each project

- PRIORITY: 0
- TAGS: feature, backlog, projects, flow, frontend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a project operator, I want a compact task board with meaningful filters and
status signals, so that I can find the next work item and inspect flow health
without scanning hundreds of indistinguishable rows.

## Steps

- [ ] Design a dense, responsive task-table/list surface using the structured
      metadata from 20260729-102158.
- [ ] Add tabs or segmented filters for open, in progress, closed, and all;
      add release/tag, priority, review verdict, and text filters.
- [ ] Show title, status, priority, scheduling tag, Flow State, review verdict,
      dependency blockers, and available artifact indicators.
- [ ] Make rows keyboard-accessible links to task detail while preserving
      native open-in-new-tab behavior.
- [ ] Add sorting, empty/loading/error states, URL-persisted filters, and
      virtualized or paged rendering if measured performance requires it.
- [ ] Add Vitest interaction coverage and Playwright coverage using the
      large mixed-state project fixture.

## Definition of Done

- A user can isolate open, blocked, unreviewed, or release-scoped work
  (test: `project-task-board.spec.ts`).
- Every task row exposes lifecycle and review state accessibly
  (test: `project-task-board.test.ts`).
- Filters survive reload and back/forward navigation
  (test: `project-task-filter-history.spec.ts`).
- The 1,000-task fixture remains responsive without layout overflow
  (test: `project-task-board-large.spec.ts`).
- The Scufris backlog is faster to understand here than through the
  current unfiltered tatr listing (manual: user check).

## Notes

- Epic: 20260729-102157.
- Depends on: 20260729-102158 and 20260729-102152.
- Keep the surface operational and information-dense; avoid dashboard cards
  for individual status categories.
