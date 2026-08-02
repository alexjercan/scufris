# Add an agent run activity timeline and hierarchy view

- PRIORITY: 63
- TAGS: feature, v0.2.0, agents, observability, frontend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As an operator of several specialized agents, I want a durable activity
timeline showing delegation, tools, approvals, artifacts, and outcomes, so that
I can understand what happened without reconstructing it from chat fragments
and server logs.

## Steps

- [ ] Define a typed run-event model for launch, parent/child relation, prompt
      summary, status transition, tool invocation, approval, artifact, token/
      cost data, outcome, cancellation, and error. Keep technical run activity
      distinct from the semantic conversation projection decided by
      20260729-220835, while giving both stable correlation references.
- [ ] Persist events in order with stable run and correlation IDs, restart
      recovery, idempotent append, pagination, and bounded retention.
- [ ] Expose project-, agent-, and run-scoped activity endpoints with filters.
- [ ] Build an unframed timeline/tree view that distinguishes hierarchy from
      chronology and links to agents, tasks, tools, and artifacts.
- [ ] Add live updates without losing history or duplicating settled events.
- [ ] Add simultaneous-agent, reconnect, pagination, missing-parent, keyboard,
      and high-volume integration tests.

## Definition of Done

- A parent run with concurrent children reconstructs the same hierarchy after
  restart (test: `test_run_activity_hierarchy_survives_restart`).
- Replaying or reconnecting a producer cannot duplicate a durable event, and
  every event can carry task/project plus optional conversation correlation
  without storing chat text as run activity
  (test: `test_run_activity_idempotent_correlation`).
- Tool, approval, artifact, cancellation, and error events are attributable to
  the correct run (test: `test_run_activity_event_attribution`).
- The browser updates live and reconnects without duplicates
  (test: `agent-run-timeline.spec.ts`).
- A failed multi-agent run can be diagnosed from the activity view
  without reading raw server logs (manual: user check).

## Notes

- Epic: 20260729-102157.
- Depends on: 20260729-102147 for durable storage and 20260729-102152 for
  browser coverage.
- Depends on: 20260729-220835 for the accepted boundary between the semantic
  conversation, technical activity, enforcement audit, and provider transcript.
- This event model should become the observability substrate for plugins,
  research swarms, approvals, and the future actor-aware orchestrator.
- V0.2.0 readiness role: provide one durable run history so the future
  conversation and Projects page link to activity rather than inventing
  parallel execution logs.
