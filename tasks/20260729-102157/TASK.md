# EPIC: Make the Project page the surface that operates the flow

- PRIORITY: 90
- TAGS: goal, epic, v0.2.0, projects, flow
- KIND: EPIC
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Epic

Make the Project page the surface the operator drives `$flow` from: select a
task, read authoritative lifecycle state, launch the legal next stage, follow
the assigned agent, inspect artifacts, approve the human stop gates, and reach
land - without opening the repository by hand.

This is an OPERATING surface, not an inspection surface. A board that only
renders tatr state does not satisfy this epic.

Authority stays with tatr. Scufris stores assignments and observations, asks
`tatr flow -n` before every transition, and renders the reason when a move is
illegal.

- Spike: tasks/20260729-220835/SPIKE.md
- Decision: tasks/20260729-220835/DECISION.md, sections 2 and 5.
- Interaction model: tasks/20260729-220835/mockup.html.

Those records are not restated here. Read them before planning any child.

## Done Means

1. Project task data includes lifecycle, scheduling, flow state, review verdict,
   dependencies, and available artifacts without parsing display-oriented CLI
   text (test: `test_project_tasks_expose_flow_metadata`).
2. Every stage launch and transition passes one server-side guard that re-reads
   the authoritative record, probes `tatr flow -n`, and requires an `operator`
   approval event for the stop gate (test: `test_flow_guard_refuses_with_reason`).
3. An unavailable action carries the reason it is unavailable; no unexplained
   disabled control exists on the page (test: `test_next_action_carries_reason`).
4. TASK, SPIKE, DECISION, REVIEW, RETRO, and NOTES records can be opened safely
   inside Scufris (test: `task-artifact-viewer.spec.ts`).
5. The workspace, the pending stop gate, and the active run survive a refresh
   and an application restart (test: `test_workspace_recovers_after_restart`).
6. manual: the Scufris repository itself is practical to OPERATE from the
   Project page for a full task lifecycle.

## Child Tasks

- [ ] 20260729-102158 (p100, v0.2.0) enrich the project task API with lifecycle
      and artifact metadata
- [ ] 20260729-102159 (p0, backlog) build a filterable flow task board for each
      project
- [ ] 20260729-102200 (p0, backlog) add an in-app task artifact viewer

Dropped from this epic in the 2026-08-03 re-cut, now `backlog`:

- 20260729-102202 responsive and accessibility findings - written against
  pages this rewrite unlinks.
- 20260729-102203 agent run activity timeline - observability, not on the
  critical path to an operating surface.

Still to be created, per `tasks/20260801-154211/TASK.md`: the task-detail
workspace, the server-side flow guard, durable assignments, the four operator
stop gates, and restart-safe recovery.

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
- V0.2.0 prerequisite slice (2026-07-29 orchestrator readiness review):
  20260729-102158, 20260729-102203, and 20260729-102202 are pulled forward to
  establish the structured task boundary, durable run observability, and a
  sound responsive/accessibility baseline. The task board and artifact viewer
  remain implementation work for the future actor-aware orchestrator epic, so
  this container stays OPEN and tagged `backlog`.
- Headline v0.2.0 epic (2026-08-03 re-cut): the maintainer cut the intervening
  polish release and went straight at the target architecture. This epic is
  promoted out of `backlog` to lead v0.2.0, restated as an operating surface,
  and its polish and observability children are demoted. It follows the
  demolition and schema tasks scheduled by 20260801-154211; it does not start
  before the new store exists.
