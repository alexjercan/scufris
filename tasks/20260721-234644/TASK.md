# U5: hidden-default polish - wordmark link, hide orchestrator from list, multi-session section, nav

- STATUS: CLOSED
- PRIORITY: 42
- TAGS: agents, frontend, spike
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

The hidden-default UX polish that makes the unified surface feel right.

- Header wordmark ("SCUFRIS / scuffed jarvis") becomes a LINK to `/` (the landing
  orchestrator chat). `web/src/_header.html` `.brand` is a plain div today.
- Hide the orchestrator from the `/agents` list in the UI (frontend guard over
  U1's server-side exclude - belt and suspenders); it is reached via `/`.
- The orchestrator-only extra section on its settings page for its multi-session
  powers (the session switcher), shown only for the orchestrator (if not folded
  into U3).
- Nav tidy: confirm "Agent" (`/`) vs "Agents" (`/agents/`) vs "Settings"
  (`/settings`) still read right and the active-state logic does not mis-highlight
  a per-agent `/agents/<id>/settings` deep link.

## Steps (/plan)

- [ ] `_header.html`: wrap the `.brand` wordmark in an `<a href="<%= basePath %>">`
      so clicking "SCUFRIS / scuffed jarvis" returns to the landing (`/`). Style
      the anchor to inherit (no underline/link color). Nav active-state already
      correct (`/settings` deep link doesn't mis-highlight - verified: `/` link is
      exact-match, `/agents/<id>/settings` falls under Agents).
- [ ] `agents-view.ts`: filter the reserved orchestrator out of the `/agents`
      list (belt-and-suspenders over U1's server-side exclude) - the orchestrator
      is reached via `/`, never the grid.
- [ ] `agent-settings-view.ts`: the orchestrator-only multi-session section. Add
      `sessions` to the global load (`/api/agent/sessions`) and render a compact
      read-only "Sessions" panel (count + current session) with a link to `/` to
      switch/manage them (the switcher itself lives on the landing chat). Shown
      only for the orchestrator (it alone is multi-session).
- [ ] Tests: the header anchor href; agents-view excludes an orchestrator row even
      if present; the orchestrator settings shows the Sessions panel, a project
      agent does not. Web `npm run ci` green.

## Definition of Done

- The header wordmark is a link to `/` (test: the anchor href="/"; manual: click
  returns to landing).
- The `/agents` list never shows the orchestrator (test: filtered even if the API
  returned it).
- The orchestrator's settings page shows a Sessions section (count + current +
  a manage-on-chat link); a project agent's does not (test).
- Full web suite green.
- manual: the wordmark link works; the orchestrator is absent from /agents; the
  Sessions section appears on /settings.

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation U5). Depends on U1 + U3 (both CLOSED). The last task; small
  polish. Full session SWITCH/new/delete stays on the landing chat sidebar (it is
  already there) - the settings Sessions panel is a read-only overview + link.
