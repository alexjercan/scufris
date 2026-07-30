# Add an in-app task artifact viewer

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,projects,flow,frontend
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a project operator, I want to read a task's planning, decision, review,
retro, and notes records in Scufris, so that the reasoning behind work is one
click away from its status.

## Steps

- [ ] Add a task-detail route and artifact navigation for TASK, SPIKE,
      DECISION, REVIEW, RETRO, and NOTES records returned by 20260729-102158.
- [ ] Render Markdown with the existing safe DOM construction approach or a
      sanitized parser that cannot execute HTML, scripts, or unsafe links.
- [ ] Add a table of contents, source toggle, copy-path/open-external actions,
      missing-artifact state, and links between referenced task IDs.
- [ ] Preserve readable code blocks, checklists, tables, findings, verdicts,
      and long records at desktop and mobile widths.
- [ ] Add hostile-content, traversal, large-file, malformed-Markdown, keyboard,
      and browser-history tests.

## Definition of Done

- All supported task records render and navigate without leaving the project
  context (test: `task-artifact-viewer.spec.ts`).
- Raw HTML, scripts, event handlers, unsafe URLs, and traversal attempts cannot
  execute or escape the project (test: `task_artifact_hostile_content.test.ts`).
- Direct URLs and back/forward navigation restore the selected artifact
  (test: `task-artifact-history.spec.ts`).
- Long SPIKE and REVIEW records remain comfortable to read (manual: user check).

## Notes

- Epic: 20260729-102157.
- Depends on: 20260729-102158, 20260729-102159, and 20260729-102152.
- This is a read-only task-record viewer, not the generic artifact editor from
  epic 20260729-102210.
