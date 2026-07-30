# Agent: read-only settings/config view + nicer tool presentation

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: feature,agent,ui,config
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

A read-only settings/config view so the user can see and understand their setup:
backend (app_server/exec/mock), model, auth mode, sandbox (read-only), whether
tools are enabled, and the configured MCP servers - plus the available tools
rendered as proper cards (name, description, source server) rather than the
current bare name+description list. This is the new, nicer home for the tools
moved off the chat head.

Read-only for now; editing settings / switching the LLM is explicitly deferred to
a later spike.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- User feedback: "maybe a settings page to enable/disable things, changing the LLM
  ... but read-only for now" and "we should see the tools in a nicer way."
- Likely needs a small `/api/agent/config` (or extend `/api/agent/info`)
  aggregating the `config.py` knobs (agent_backend, agent_model, agent_auth_mode,
  agent_tools_enabled, mcp_servers; sandbox is always read-only).
- Decide page-vs-panel at /plan (a new `/settings/` nav page reuses the multipage
  webpack pattern - see lesson `webpack-multipage-htmlplugin-per-page`). Escape
  everything; keep render side-effect-free for jsdom.

## Decision

Built as a new `/settings/` NAV PAGE (not a panel): the user said "settings page",
it reuses the existing multipage webpack pattern (Agent/Stats), and it is the
natural home for the future editable-settings work.

## Implementation

- Backend (`app.py`): `AgentConfig` + `McpServerInfo` models and a
  `GET /api/agent/config` endpoint aggregating the read-only knobs (enabled,
  backend, model, auth_mode, tools_enabled, sandbox="read-only", and the MCP
  servers: the built-in `scufris` when tools are on, plus any configured ones).
  Reuses the existing `GET /api/agent/tools` for the tool cards.
- Frontend: `settings.html` + `settings.ts` entry + a side-effect-free
  `settings-view.ts` (`renderSettings(root, config, tools)` builds the DOM;
  `startSettings()` fetches + renders, with a fallback on error). Three cards:
  Agent (config rows), MCP servers (id + built-in/configured badge), Tools
  (name/description cards in an auto-fill grid). All values escaped.
- Wiring: webpack `settings` entry + HtmlWebpackPlugin (`settings/index.html`) +
  a `historyApiFallback` rewrite; a "Settings" nav link in `_header.html`.
  `StaticFiles(html=True)` serves `/settings/` with no backend change (same as
  `/stats/`). Styles for `.settings*` and `.tool-card*`.

## Tests / verification

- `settings-view.test.ts` (5): renders config + servers + tool cards; disabled
  agent + empty tools; null-config fallback; hostile tool name/description is
  escaped (no `<img>`/`<script>`); clean re-render (no duplicate cards).
- `test_app.py` (2): `/api/agent/config` reports the effective settings and the
  built-in + configured servers; omits the built-in server when tools are off.
- E2e serve+curl (per `frontend-verify-needs-e2e-serve`): the real backend serves
  `/settings/` (200, title + `#settings` + nav link) and `/api/agent/config`
  returns the expected JSON. 131 pytest + 78 frontend green.

## Note for the coupled task

This is the new home for the tools; task 20260720-102600 (chat head redesign) can
now drop the head's `tools` toggle and point users here.
