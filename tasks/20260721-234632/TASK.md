# U4: routing/entries so / == orchestrator and /agents/<id>[/settings] share the components

- PRIORITY: 44
- TAGS: agents, frontend, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Wire the routing so `/` and `/settings` are the ORCHESTRATOR's chat + settings and
`/agents/<id>` and `/agents/<id>/settings` are a project agent's - all using the
SAME components, differing only by the resolved agent id.

- `/settings` entry mounts the unified settings (U3) with agent id =
  `orchestrator`; `/agents/<id>/settings` mounts it with `agentIdFromPath` (the
  backend `/agents/{id}/{rest:path}` catch-all already serves the shell - confirm,
  add the entry/mount that reads the `/settings` sub-path).
- `/` and `/agents/<id>` already share the chat (B5d) - confirm nothing regresses.
- Retire the per-agent settings modal + its toggle now that settings is a page.
- Add a "Settings" affordance on each agent's page linking to
  `/agents/<id>/settings`.

## Steps (/plan)

- [x] `settings-view.ts`: EXPORT the reusable global-section renders
      (`renderServerControls` MCP, `renderToolControls` + the tool grid,
      `renderProfileSwitcher`, `renderPanels`) + `SettingsActions`/`SettingsExtras`,
      and add a slim `renderGlobalToggles(config, actions)` (agent_enabled + tools
      toggles ONLY - backend/model/permission_mode now live in the agent-settings
      form, so they are NOT duplicated here).
- [x] `agent-settings-view.ts`: add an optional `global` dep to
      `AgentSettingsDeps` ({ config, tools, extras, actions } - present only for
      the orchestrator). When present, `renderAgentSettings` appends the GLOBAL
      sections after the panels: global toggles + MCP servers + tools + profiles
      (the shared agent config). A project agent has no `global` -> just its
      fields/health/panels.
- [x] `/settings` (settings.ts entry): mount `createAgentSettings` for the
      ORCHESTRATOR with the `global` dep wired to `/api/agent/config` + `/tools` +
      the console panels + the config actions - so `/settings` IS the
      orchestrator's settings page, same component as `/agents/<id>/settings`.
      Retire `settings-view.startSettings` + the `renderSettings` top-level
      composition (keep the exported section renders it reuses).
- [x] `agent-detail.ts` path-branch: when the settings page is the ORCHESTRATOR
      (`agentIdFromPath === orchestrator`), also wire the `global` dep so
      `/agents/orchestrator/settings` === `/settings`. Confirm `/` is unchanged
      (already the orchestrator chat).
- [x] Tests: `/settings` renders the orchestrator's unified settings + the global
      sections (toggles/MCP/tools/profiles); a project agent's page has NO global
      sections; the global toggles + MCP add/remove + profile switch still work;
      retire/port the `settings-view` composition tests. Web `npm run ci` green.

## Definition of Done

- `/settings` renders the orchestrator's settings via the SAME
  `createAgentSettings` component as `/agents/<id>/settings`, plus the global
  sections (tools/MCP/profiles/enabled) - which appear ONLY for the orchestrator
  (test: a project agent's settings has no global sections).
- `settings-view.ts` no longer exposes a top-level `renderSettings`/`startSettings`
  page composition (its section renders survive as reused pieces)
  (grep: `startSettings` gone / the entry uses createAgentSettings).
- `/` remains the orchestrator chat; backend/model appear ONCE (the agent form),
  not duplicated by a global control.
- Full web + backend suites green.
- manual: `/`, `/settings`, `/agents/<id>`, `/agents/<id>/settings` all work and
  `/settings` == `/agents/orchestrator/settings`.

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation A1 / U4). Depends on U3 (CLOSED).
- Adopts U3's deferred R3/R5: the read-only path gets wired here (config.writable),
  and the orchestrator health stays global (it IS the system agent).
