# Settings backend: editable tools (per-tool enable/disable + MCP server add/remove)

- PRIORITY: 42
- TAGS: feature, agent, backend, mcp
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want to enable/disable individual agent tools and add or
remove MCP servers from the settings page, so I can shape what the agent can do
without touching config files. Today `agent_tools_enabled` is a single global
switch and `mcp_servers` is env-only JSON.

## Steps

- [x] Add a `disabled_tools: list[str]` field to `Settings` (env
      `SCUFRIS_DISABLED_TOOLS`, default empty), on the task-1 override
      whitelist (+ `AgentConfigUpdate`) so it is editable + persisted.
- [x] Filter the built-in Scufris MCP tools by `disabled_tools` at the REAL
      enforcement point: the agent passes the set to the scufris server via
      codex's per-server env (`_server_override` gained an `env` param;
      `_mcp_overrides` sets `mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS`),
      and `mcp_server.apply_disabled_tools` removes them from the FastMCP
      registry at startup (via `_tool_manager.remove_tool`) so codex never sees
      them - not just hidden in the UI. (Probed live that codex accepts
      `mcp_servers.<id>.env.KEY`.)
- [x] Reflect enabled/disabled state in `GET /api/agent/tools` via a new
      `AgentTool.enabled` (= name not in `settings.disabled_tools`).
- [x] Make `mcp_servers` editable through the task-1 write endpoint (already in
      the whitelist). Added boundary validation in `patch_agent_config`: a bad
      id or empty command -> 422; the built-in `scufris` id is reserved.
      (Amended: kept `McpServerSpec` PERMISSIVE - the strict validator was
      reverted because the repo's pattern is permissive construction + skip in
      `_mcp_overrides`, and a strict model would crash startup on a bad env
      entry; validation lives at the user-facing endpoint instead.)
- [x] Tests: `test_apply_disabled_tools_removes_and_reports` (registry removal
      + global-restore fixture), `test_mcp_overrides_passes_disabled_tools_env`,
      `test_tools_endpoint_reports_enabled`, `test_patch_disabled_tools_persists`,
      `test_add_mcp_server_persists`, `test_add_mcp_server_rejects_bad_id`.
- [x] `.env.example`: documented `SCUFRIS_DISABLED_TOOLS`.

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

## Close-out

- Enforcement design: codex registers whole MCP servers, not individual tools,
  so per-tool disable had to happen INSIDE the scufris server. The agent injects
  `SCUFRIS_DISABLED_TOOLS` (comma-separated) via codex's per-server `env`
  config; `mcp_server.main()` calls `apply_disabled_tools`, which
  `remove_tool`s each from the FastMCP registry before serving. A disabled tool
  is therefore never advertised or callable - the UI `enabled` flag is just a
  mirror, not the guard. Probed `codex mcp list -c mcp_servers.x.env.KEY=...`
  live first to confirm codex accepts per-server env (the Env column populated).
- Test hazard: `apply_disabled_tools` mutates the process-global `mcp`
  singleton (fine in the real fresh-per-spawn subprocess), so its tests use a
  `restore_tool_registry` fixture that snapshots and restores
  `_tool_manager._tools` - otherwise the removal leaks into every later test
  that lists tools (the persistent-global-state test-reset lesson).
- Decision to keep `McpServerSpec` permissive: adding a strict id/command
  validator broke `test_mcp_overrides_skips_invalid_or_reserved_id` and would
  crash startup on a bad `SCUFRIS_MCP_SERVERS` env entry. The repo's existing
  pattern is permissive-construct + skip-in-overrides; validation for a USER
  add lives at the endpoint (clear 422) instead. Reverted the model validators.
- `mcp_servers` add/remove needed no new store work - T1's whitelist already
  covered it; this task only added the endpoint-boundary id/command check.
- Ran `python -m pytest` throughout (T1's lesson) so the worktree source loaded.
- Self-reflection: the strict-validator detour cost one edit-revert cycle; the
  existing skip test flagged it fast. Reading that test BEFORE adding the
  validator would have avoided the round-trip - a reminder to grep the existing
  tests of a model before tightening it.
