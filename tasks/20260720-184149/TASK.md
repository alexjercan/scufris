# Settings UI: profile switcher + informative panels (sessions/usage/context/memory/account)

- STATUS: CLOSED
- PRIORITY: 28
- TAGS: feature,agent,ui

## Story

As the operator, I want the console to show the richer informative panels
(sessions, usage/quota, context, memory footprint, account) and a profile
switcher, so the settings page becomes a real operator dashboard, not just a
config list.

## Steps

- [x] Five read-only panels (`renderPanels` -> `infoPanel`): Sessions
      (count+current), Usage (plan/used/window), Context (window-fill/turns/
      tool-calls), Memory (sessions/on-disk/newest), Account (auth/model/
      status). Each row shows `-` when its datum is null (fetch failed or agent
      off), so a panel never collapses.
- [x] Profile switcher (`renderProfileSwitcher`, writable only): lists profiles
      with the active one marked (its activate button disabled), activates via
      `activateProfile` -> `POST .../activate` and reloads, a delete button per
      non-active profile (confirmed), and a "save as" form -> `createProfile`.
- [x] `renderSettings` stays pure: panels come from an optional `SettingsExtras`
      bundle and the switcher from `SettingsActions`; `startSettings` fetches
      the six extras best-effort (a failed one -> null -> that panel degrades)
      and wires the profile actions.
- [x] jsdom tests: panels-with-data, panels-degrade-to-dash, panels-omitted-
      without-extras, switcher-marks-active, activate-on-click, switcher-hidden-
      when-read-only, create-from-save-as-form.

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

## Close-out

- Pure-additive frontend task on top of T3/T4/T5: no backend change. Extended
  the existing `SettingsActions` seam with profile ops and added a
  `SettingsExtras` bundle so `renderSettings` stays pure and jsdom-testable
  (existing display tests were unaffected; only the fake-actions factory grew).
- Degraded states follow `stable-rows-with-dash-beats-conditional-sections`:
  `infoPanel` always renders every row and shows `-` for a null value, so a
  panel with no data does not collapse or jump. `startSettings` fetches the six
  extras with a `maybe()` best-effort wrapper (a failed fetch -> null -> that
  one panel degrades, the page still renders).
- Verified per `frontend-verify-needs-e2e-serve`: served the built bundle
  through uvicorn (mock backend) and confirmed `/settings/` 200, profiles
  list/create/activate, memory footprint (27 real sessions), and account all
  return live data. Isolated + cleaned the state dir.
- No `git add -A` this time (last task's lesson): staged explicit paths.
- Self-reflection: smooth - the pure-render + injected-seam design from T5 made
  T6 almost mechanical. The only care point was targeting the profile "save as"
  form vs the MCP add-server form in tests (both use `.settings__addserver`);
  selected by the input's aria-label instead of the shared class.
