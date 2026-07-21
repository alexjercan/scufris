# U5: hidden-default polish - wordmark link, hide orchestrator from list, multi-session section, nav

- STATUS: OPEN
- PRIORITY: 42
- TAGS: agents,frontend,spike

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

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation U5). Depends on U1 + U3. Mostly small, user-eyeballed polish.
