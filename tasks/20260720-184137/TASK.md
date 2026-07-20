# Settings backend: editable tools (per-tool enable/disable + MCP server add/remove)

- STATUS: OPEN
- PRIORITY: 42
- TAGS: feature,agent,backend,mcp

## Story

As the operator, I want to enable/disable individual agent tools and add or
remove MCP servers from the settings page, so I can shape what the agent can do
without touching config files. Today `agent_tools_enabled` is a single global
switch and `mcp_servers` is env-only JSON.

## Steps

- [ ] Add a `disabled_tools: list[str]` field to `Settings` (env
      `SCUFRIS_DISABLED_TOOLS`, default empty), on the task-1 override
      whitelist so it is editable + persisted.
- [ ] Filter the built-in Scufris MCP tools by `disabled_tools` where the agent
      registers/derives them, so a disabled tool is not offered to codex. Grep
      for where `mcp` tools flow to codex (`scufris/agent.py` MCP overrides,
      `scufris/mcp_server.py`); confirm the enforcement point actually removes
      the tool from the turn, not just the UI list.
- [ ] Reflect enabled/disabled state in `GET /api/agent/tools` (add an
      `enabled: bool` to `AgentTool`) so the UI can show toggles.
- [ ] Make `mcp_servers` editable through the task-1 write endpoint: validate
      each `McpServerSpec` (id regex `^[A-Za-z0-9_]+$`, non-empty command),
      apply+persist. Adding/removing a server updates what codex registers on
      the next turn.
- [ ] Tests: a disabled tool is excluded from the agent's registered tool set
      (not just the API list); toggling it back re-includes it; adding an MCP
      server via the endpoint persists and appears in `GET /api/agent/config`;
      an invalid server id is rejected.

## Definition of Done

- A tool in `disabled_tools` is absent from what the agent exposes to codex,
  and clearing it restores the tool (test: `disabled_tool_not_registered`).
- `GET /api/agent/tools` reports each tool's `enabled` state
  (test: `tools_endpoint_reports_enabled`).
- Adding an MCP server through the write endpoint persists and shows in the
  effective config; an invalid id is rejected
  (test: `add_mcp_server_persists`, `add_mcp_server_rejects_bad_id`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).

## Notes

- Depends on: 20260720-184136 (the override store + write endpoint). Uses
  its whitelist + persistence.
- Relevant files: `scufris/config.py`, `scufris/agent.py` (`_mcp_overrides`,
  MCP registration), `scufris/mcp_server.py` (tool registry), `scufris/app.py`
  (`AgentTool`, `get_agent_tools`, `get_agent_config`).
- Per-tool disable is enforced at the registration/override boundary, not by
  trusting the client - the tool must genuinely not reach codex.
