# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- The `tatr_ls`, `tatr_show` and `tatr_new` MCP tools. Now that the scufris server
  is orchestrator-only, the orchestrator manages tatr tasks with the `tatr` skill
  via `Bash`, so a dedicated MCP wrapper is redundant. The host/observe tools
  (`host_stats`, `disk_usage`, `list_processes`, `list_agents`, `agent_status`) and
  the new control tools remain; the tool-steering preamble no longer mentions tatr.

### Changed

- The built-in `scufris` MCP server (host/observe tools and its tool-steering
  preamble) is now ORCHESTRATOR-ONLY: it is registered for the landing
  orchestrator's turns only, not for every agent. Regular project agents no longer
  receive the scufris tools and draw their tools from their own project
  config/skills. This threads an `is_orchestrator` flag through the backend
  `stream` path; operator-declared `mcp_servers` still apply to every agent.

### Added

- Full CRUD orchestrator control tools on the scufris MCP server: `get_project`,
  `update_project`, `delete_project`, `update_agent` and `delete_agent` join the
  existing create/list/run/message tools, so the orchestrator can edit an agent's
  permission mode (manual|edit|auto), provider (codex|claude) and model, and manage
  projects, all from chat. The PATCH tools send only the fields you pass. The agent
  write tools edit REGULAR agents only - the reserved orchestrator configures itself
  via settings and is refused.
- Orchestrator control tools on the scufris MCP server (orchestrator-only): the
  landing orchestrator can now DO dashboard actions, not just observe. `list_projects`,
  `create_project`, `create_agent`, `run_agent` and `message_agent` call the
  dashboard's own HTTP API at `SCUFRIS_API_BASE` (127.0.0.1:<port>, injected when the
  dashboard spawns the server), reusing each endpoint's validation and lifecycle since
  the MCP subprocess cannot touch the live in-app supervisor. Curated and bounded like
  the existing tools; a non-2xx or network failure returns `error:` text, never an
  exception.
- Settings page: an interactive "try it" runner on each enabled tool card - reveal
  a form generated from the tool's parameter schema, confirm, and run one MCP tool
  in isolation with its result rendered inline, without a chat turn. Backed by a new
  `POST /api/agent/tools/{name}/run` endpoint that runs a single scufris tool
  in-process (bypassing the agent) and refuses a disabled tool (403), an unknown tool
  (404), or bad args (422). The tools listing (`GET /api/agent/tools`) now also
  exposes each tool's typed parameter schema.
