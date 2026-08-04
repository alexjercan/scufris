# U3: unified settings PAGE component for all agents (replaces settings-view + the modal)

- PRIORITY: 46
- TAGS: agents, frontend, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Build ONE unified settings PAGE component that renders any agent's settings, so
the orchestrator and project agents share the exact same settings UI. Replaces
BOTH the `/settings` page's `settings-view.ts` and the per-agent settings MODAL
(`agent-detail-view.renderSettingsModal`).

`createAgentSettings(root, {agentId})` renders:
- the agent's EDITABLE fields via the existing `agentFields`: backend picker ->
  model auto-defaults to that backend's default; model via autocomplete;
  permission/sandbox mode (manual->edit->auto); description.
- the health card (keep it).
- the detailed panels: context (from the agent's session) + usage + memory +
  account (from U2).
- global sections (tools enabled, MCP servers, profiles) placed per the SPIKE
  open question (recommend: editable on the orchestrator's page, linked from a
  project agent's).
- an orchestrator-ONLY extra section for its multi-session powers (session
  switcher) - may be split to U5.

Pure render + injected deps so jsdom drives it; a real PAGE, not a modal.

## Steps (/plan)

Scope re-cut (avoids throwaway shims): U3 delivers the per-agent settings PAGE end
to end via the SHARED detail shell's path-branch; the orchestrator-at-root
symmetry (`/`, `/settings`) + retiring `settings-view.ts` + folding the GLOBAL
sections (tools/MCP/profiles) onto the orchestrator's page is U4.

- [x] `web/src/agent-settings-view.ts`: `renderAgentSettings(root, data, deps)`
      (PURE) + `createAgentSettings(root, deps)` (load->render->wire save/reload) +
      `startAgentSettings()` (entry, real endpoints, `agentIdFromPath`). Composes:
      back link + agent name; the EDITABLE fields form via the existing
      `agentFields` (backend picker -> auto model default, model autocomplete,
      permission mode, description) saving `PATCH /api/agents/{id}` (the orchestrator
      routes to settings via U1); the HEALTH card; and the detailed PANELS -
      status/context (from `/api/agents/{id}/status`), usage/memory/account (from
      U2's `/api/agents/{id}/{usage,memory,account}`).
- [x] Reuse, don't duplicate: export `renderHealthCard` (+ its `healthRow`) from
      `settings-view.ts` and reuse it; build the compact per-agent panel boxes with
      the shared `settings__card`/`usage-block` styling.
- [x] Route the per-agent settings PAGE without a new shell/entry: the backend
      catch-all already serves `agent-detail.html` for `/agents/<id>/settings`. Add
      an `#agent-settings` container to `agent-detail.html`; branch `agent-detail.ts`
      on the path - `/agents/<id>/settings` -> `startAgentSettings` (no chat);
      `/agents/<id>` -> the chat detail as today. The detail sidebar's "Settings"
      button becomes a LINK to `/agents/<id>/settings`; retire
      `renderSettingsModal` + the modal wiring. The orchestrator now gets a Settings
      link too (U1 made it editable).
- [x] Tests (`agent-settings-view.test.ts`): `renderAgentSettings` shows the
      fields form + health + each panel for a project agent AND the orchestrator;
      save reads the fields and PATCHes; read-only server hides the form / shows
      read-only rows; hostile strings escaped. Update `agent-detail-view.test.ts`
      for the modal removal + the Settings link. Port the meaningful modal tests.
- [x] Full web `npm run ci` green (webpack build is the type gate) + backend
      pytest unaffected.

## Definition of Done

- One `createAgentSettings` component renders any agent's settings (editable
  fields + health + context/usage/memory/account panels); no per-agent settings
  MODAL remains (grep: `renderSettingsModal` gone from agent-detail-view).
- `/agents/<id>/settings` shows the settings page for a project agent and the
  orchestrator; the detail page's Settings button links to it
  (test + manual: the page renders and a save persists).
- Full web + backend suites green.
- manual: `/agents/<id>/settings` shows the fields + health + panels and editing
  a field saves.

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation C1). Depends on U1 + U2 (both CLOSED). The big frontend slice.
- U4 owns: `/` and `/settings` mount the SAME chat + settings for the orchestrator
  (the root-exposure symmetry) and retire `settings-view.ts`, folding the GLOBAL
  sections (tools/MCP/profiles) onto the orchestrator's settings page (spike:
  those are shared, so they live on the orchestrator, not every agent).
