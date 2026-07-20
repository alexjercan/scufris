# Settings UI: profile switcher + informative panels (sessions/usage/context/memory/account)

- STATUS: OPEN
- PRIORITY: 28
- TAGS: feature,agent,ui

## Story

As the operator, I want the console to show the richer informative panels
(sessions, usage/quota, context, memory footprint, account) and a profile
switcher, so the settings page becomes a real operator dashboard, not just a
config list.

## Steps

- [ ] In `web/src/settings-view.ts`, add read-only panels fed by the existing
      + new endpoints: a Sessions summary (count + current, from
      `/api/agent/sessions`), Usage/quota (`/api/agent/usage`), Context
      (`/api/agent/context`), Memory footprint (`/api/agent/memory`), Account
      (`/api/agent/account`). Each panel degrades gracefully (shows `-` /
      "unavailable") when its fetch fails or the agent is disabled.
- [ ] Add a profile switcher control (list from `/api/agent/profiles`, active
      highlighted) that activates a profile via
      `POST /api/agent/profiles/activate` and re-renders the whole page from
      the new effective config; plus save-as / rename / delete affordances.
- [ ] Keep `renderSettings` pure and jsdom-testable; fetch orchestration stays
      in `startSettings`.
- [ ] jsdom tests: each panel renders its data and its empty/degraded state;
      the profile switcher lists profiles and marks the active one; switching
      triggers the activate call (mock fetch).

## Definition of Done

- All five panels render real data on the running app and a graceful empty
  state when unavailable (test: `settings_panels_render_and_degrade`; manual:
  each panel shows real data on the running server).
- The profile switcher lists profiles, marks the active one, and switching
  re-renders the page from the new config (test:
  `profile_switcher_activates`; manual: switch a profile, page reflects it).
- `npm run ci` passes in `web/` (cmd: `cd web && npm run ci`).
- End-to-end: serve the built bundle through the backend and confirm the
  panels + switcher load (manual: `/settings` renders live).

## Notes

- Depends on: 20260720-184138 (profiles endpoints) and 20260720-184146 (memory/account endpoints);
  softly on 20260720-184148 (shared page/controls, land after to avoid churn).
- Lessons: `frontend-verify-needs-e2e-serve` (serve through backend, not just
  a green build), `flex-display-defeats-the-hidden-attribute` and
  `stable-rows-with-dash-beats-conditional-sections` for the degraded states.
