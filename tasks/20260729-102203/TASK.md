# Add an agent run activity timeline and hierarchy view

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,agents,observability,frontend

## Story

As an operator of several specialized agents, I want a durable activity
timeline showing delegation, tools, approvals, artifacts, and outcomes, so that
I can understand what happened without reconstructing it from chat fragments
and server logs.

## Steps

- [ ] Define a typed run-event model for launch, parent/child relation, prompt
      summary, status transition, tool invocation, approval, artifact, token/
      cost data, outcome, cancellation, and error.
- [ ] Persist events in order with stable run and correlation IDs, restart
      recovery, pagination, and bounded retention.
- [ ] Expose project-, agent-, and run-scoped activity endpoints with filters.
- [ ] Build an unframed timeline/tree view that distinguishes hierarchy from
      chronology and links to agents, tasks, tools, and artifacts.
- [ ] Add live updates without losing history or duplicating settled events.
- [ ] Add simultaneous-agent, reconnect, pagination, missing-parent, keyboard,
      and high-volume integration tests.

## Definition of Done

- A parent run with concurrent children reconstructs the same hierarchy after
  restart (test: `test_run_activity_hierarchy_survives_restart`).
- Tool, approval, artifact, cancellation, and error events are attributable to
  the correct run (test: `test_run_activity_event_attribution`).
- The browser updates live and reconnects without duplicates
  (test: `agent-run-timeline.spec.ts`).
- manual: a failed multi-agent run can be diagnosed from the activity view
  without reading raw server logs.

## Notes

- Epic: 20260729-102157.
- Depends on: 20260729-102147 for durable storage and 20260729-102152 for
  browser coverage.
- This event model should become the observability substrate for plugins,
  research swarms, and approvals.

## Flow State

- FLOW STEP: PLANNING
