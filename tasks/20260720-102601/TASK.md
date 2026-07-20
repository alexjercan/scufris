# Agent: read-only settings/config view + nicer tool presentation

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui,config

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
