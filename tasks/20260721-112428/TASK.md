# F0: quick UI polish bugs (SSE reattach on select, status poll interval, empty states)

- STATUS: CLOSED
- PRIORITY: 52
- TAGS: agents,ux,frontend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED


## Goal

Quick, independent frontend polish (shippable first, no backend change):
- Reattach the SSE EventSource when SELECTING an already-running agent (today it
  only opens on Run); the events endpoint replays via Last-Event-ID.
- Add a modest status `setInterval` while a running agent is open, so
  turns/tokens refresh even between SSE events (mirror stats-view polling).
- Empty/guidance states: "create a project first" when the agent create form has
  no projects (disable submit); show "not started" instead of 0s for a never-run
  agent.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (current-state review, bugs m2/m3/m5).
- No deps; can land before the refactors.

## Steps

- [x] Reattach the SSE stream when a running agent is selected: in `pollStatus`,
      once a status arrives with state `running`/`queued` and no `EventSource` is
      open, call `openEvents(id)`. Only opening on an active run avoids the
      404-then-auto-reconnect loop of opening events on an idle agent.
- [x] Add a bounded status poll interval: `setInterval(DEFAULT_POLL_SECONDS)`
      that re-polls the SELECTED agent's status ONLY while it is running/queued,
      and SKIPS the tick when a create-form field is focused (the pure re-render
      would otherwise wipe the user's input). Refreshes turns/tokens between SSE
      events.
- [x] Empty/guidance states in `renderAgents`/`createForm`: when there are no
      projects, show "create a project first" and disable the create submit;
      show "not started" instead of literal 0s for a never-run agent
      (state idle + turns 0).
- [x] jsdom tests: create form disabled + guidance with no projects; a never-run
      agent shows "not started". (SSE/interval wiring lives in startAgents, not
      the pure render, so it stays e2e-verified; assert the render-level pieces.)
- [x] `npm run ci` green; close-out.

## Definition of Done

- With no projects, the agent create form is disabled and shows guidance
  (test: `agents_create_disabled_without_projects`).
- A never-run agent's status shows "not started", not 0s
  (test: `agents_status_shows_not_started`).
- `npm run ci` passes in web/ (cmd: `cd web && npm run ci`).
- manual: selecting a running agent reattaches its live events; the status
  refreshes on an interval without wiping the create form while typing.

## Close-out

What changed (web/src/agents-view.ts):
- SSE reattach: `pollStatus` now opens the event stream when a selected agent's
  status is `running`/`queued` and none is open - so selecting an already-running
  agent reattaches its live events. Guarded to only open on an ACTIVE run, which
  avoids opening an EventSource on an idle agent (a 404 that would auto-reconnect).
- Bounded status interval: a `setInterval(DEFAULT_POLL_SECONDS)` re-polls the
  selected agent's status ONLY while it is active, and SKIPS the tick when a
  create-form field is focused (`INPUT/TEXTAREA/SELECT` inside the root) so the
  pure re-render never wipes the operator's input.
- Empty/guidance states: the create form's submit is disabled with a "create a
  project first" note when there are no projects; a never-run agent (idle + no
  session) shows "not started - run this agent to begin." instead of literal 0s.
- Tests: `disables the create form ... when there are no projects`, `shows 'not
  started' ... for a never-run agent` (the SSE/interval wiring lives in
  startAgents and is e2e-verified, not jsdom-tested).

Result: `npm run ci` green (135 tests, +2), webpack builds.

Self-reflection: the status-interval had a real trap - the pure full-re-render
would wipe the create form during typing. Bounding the interval to active runs +
a focus-guard was the fix; the cleaner answer (status on its own page, no form)
arrives with the F3 detail page.
