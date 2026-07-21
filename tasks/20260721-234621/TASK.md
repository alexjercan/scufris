# U3: unified settings PAGE component for all agents (replaces settings-view + the modal)

- STATUS: OPEN
- PRIORITY: 46
- TAGS: agents,frontend,spike

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

## Notes
- EPIC/umbrella: tasks/20260721-234126. Spike: tasks/20260721-234433/SPIKE.md
  (recommendation C1). Depends on U1 + U2. The big frontend slice.
