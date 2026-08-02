# Automate critical desktop and mobile user journeys

- PRIORITY: 66
- TAGS: testing, v0.2.0, frontend, e2e, ui
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a user, I want Scufris's primary workflows continuously exercised exactly
as I use them, so that a green build proves the application is clickable,
responsive, persistent, and understandable in a browser.

## Steps

- [ ] Cover navigation across home, stats, agents, projects, settings, agent
      detail/settings, and project detail at desktop and mobile widths.
- [ ] Cover project registration, agent creation, editing, opening, deletion,
      and expected confirmation behavior.
- [ ] Cover chat streaming, tool events, stop/cancel, controlled errors,
      transcript reload, session continuity, and export.
- [ ] Cover the existing bidirectional agent loop over public APIs and visible
      surfaces: concurrent agents, `request_input`, `report_back`, pending
      attribution, acknowledge, and restart recovery.
- [ ] Cover task filtering and artifact navigation when the operator-workspace
      tasks land, without coupling the initial suite to unfinished features.
- [ ] Assert focus behavior, usable touch targets, no horizontal scroll,
      coherent loading/empty/error states, and no clipped dynamic text.
- [ ] Add stable visual snapshots only for high-value layouts and document the
      intentional-update workflow.

## Definition of Done

- Project and agent creation through the UI work and persist across reload
  (test: `project-and-agent-lifecycle.spec.ts`).
- Chat stream, cancellation, error recovery, transcript reload, and export work
  through visible controls (test: `agent-chat-lifecycle.spec.ts`).
- Bidirectional multi-agent outcomes retain the correct agent/run attribution
  across reload (test: `multi-agent-outcome-lifecycle.spec.ts`).
- All routes pass the mobile navigation, focus, overflow, and axe assertions
  (test: `mobile-user-journeys.spec.ts`).
- The suite observes no unexpected `4xx`, `5xx`, console error, or page error
  (cmd: `cd web && npm run test:e2e`).

## Notes

- Epic: 20260729-102149.
- Depends on: 20260729-102152.
- Extend these journeys in the same task as future user-facing features.
- This task establishes the pre-orchestrator browser baseline; it does not test
  the future Scufris-owned conversation until that feature exists.
