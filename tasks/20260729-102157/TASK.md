# EPIC: Turn Projects into a flow-native operator workspace

- STATUS: OPEN
- PRIORITY: 0
- TAGS: goal,epic,backlog,projects,flow

## Epic

Turn the existing Projects page from a shallow tatr listing into the daily
operator surface for `$flow`: task status, release scheduling, dependencies,
plan/review/retro artifacts, agent runs, and actionable filtering should be
visible without opening the repository by hand.

## Done Means

1. Project task data includes lifecycle, scheduling, flow state, review verdict,
   dependencies, and available artifacts without parsing display-oriented CLI
   text (test: `test_project_tasks_expose_flow_metadata`).
2. A project with hundreds of tasks remains easy to scan by status, release,
   tag, priority, and text (test: `project-task-board.spec.ts`).
3. TASK, SPIKE, DECISION, REVIEW, RETRO, and NOTES records can be opened safely
   inside Scufris (test: `task-artifact-viewer.spec.ts`).
4. Agent runs are traceable as a parent/child timeline with tools, outcomes,
   approvals, and artifacts (test: `agent-run-timeline.spec.ts`).
5. manual: the Scufris repository itself is practical to inspect and operate
   from the Projects page.

## Child Tasks

- [ ] 20260729-102158 (p0, scufris) enrich the project task API with lifecycle
      and artifact metadata
- [ ] 20260729-102159 (p0, scufris) build a filterable flow task board for each
      project
- [ ] 20260729-102200 (p0, scufris) add an in-app task artifact viewer
- [ ] 20260729-102202 (p0, scufris) fix responsive layout and accessibility
      audit findings
- [ ] 20260729-102203 (p0, scufris) add an agent run activity timeline and
      hierarchy view

## Decisions

- Pending 20260729-102158: machine-readable tatr boundary and task metadata
  model.
- Pending 20260729-102200: safe Markdown rendering and artifact URL scheme.

## Manual Acceptance

- (pending) 20260729-102159: the task board is dense enough for repeated use
  without becoming visually noisy.
- (pending) 20260729-102203: run history makes multi-agent behavior easier to
  understand rather than adding another log dump.

## Sequencing

- Post-v0.1.0 order (2026-07-29 backlog review): SECOND of the five backlog
  epics, after 20260729-102149. Best odds of daily use of anything unscheduled,
  because the operator already lives in tatr - but v0.1.0 spends its budget on
  host agency, which is the differentiator this cannot claim.
- Stays `backlog` at priority 0 until pulled into the next release plan.

## Flow State

- FLOW STEP: PLANNING
