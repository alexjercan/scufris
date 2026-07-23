# Role-scoped per-agent tools endpoint + tools panel on each agent's settings page

- STATUS: CLOSED
- PRIORITY: 40
- TAGS: feature, agent, ui, backend

## Story

As an operator looking at a sub-agent, I want its tools listing to show only the
tools THAT agent actually has (a codex sub-agent: just `request_input`), not the
orchestrator's full 18-tool surface, and I want each agent's settings page to show
its available tools like the orchestrator's does - so the real tool surface is
transparent and correct.

## Context (grounded)

- BUG: `GET /api/agent/tools` (`scufris/app.py:1517-1539`, `get_agent_tools`)
  returns `await mcp.list_tools()` on the dashboard's GLOBAL, never-role-scoped
  `mcp` instance - all ~18 tools - regardless of caller. Role scoping only happens
  in a SPAWNED server via `mcp_server.apply_role` (`scufris/mcp_server.py:661-674`)
  using `_AGENT_ROLE_TOOLS = {"request_input"}` (`mcp_server.py:648`); the endpoint
  never applies it. So any per-agent tools display reads the orchestrator's set.
- The role source of truth: `ROLE_ORCHESTRATOR` keeps all tools EXCEPT
  `_AGENT_ROLE_TOOLS`; `ROLE_AGENT` keeps ONLY `_AGENT_ROLE_TOOLS`
  (`mcp_server.py:661-674`). A backend-agnostic "tools for this agent" also depends
  on whether the agent's BACKEND delivers MCP at all (codex yes; claude not until
  the claude-MCP task lands - a claude sub-agent currently has 0 scufris tools).
- Frontend: the orchestrator settings render a Tools section via
  `agent-settings-view.ts` (`agentSettingsDeps` ~line 449-451 fetches
  `/api/agent/tools` ONLY when `isOrchestrator`; section rendered ~line 337-342 via
  `renderToolControls` in `settings-view.ts:250-273`). A sub-agent's settings page
  omits the tools section entirely today. The "N tools available" count also shows
  in `agent-view.ts` (~line 38-55, 255).

## Steps

- [x] REPRODUCE first: pin the exact UI surface that shows "18 tools available" for
      a codex sub-agent (agent-view sidebar vs settings vs chat panel), so the fix
      targets the real render path. Note it in NOTES.md.
- [x] Backend: add a role-scoped tools listing keyed by agent. Prefer
      `GET /api/agents/{agent_id}/tools` (mirrors `/api/agents/{id}/health`,
      `app.py:1457`): resolve the agent, compute its role (orchestrator vs agent),
      and return the role-scoped tool set. Factor a helper
      `tools_for_role(role) -> list[AgentTool]` reusing `_AGENT_ROLE_TOOLS`
      /`apply_role` logic (single source of truth with mcp_server). Be truthful
      about backend: a sub-agent whose backend has no scufris MCP (claude, today)
      returns [] - not `request_input`. A codex sub-agent returns `[request_input]`;
      the orchestrator returns its full set.
- [x] Keep `/api/agent/tools` working for the orchestrator (or route it through the
      same helper for `ORCHESTRATOR_ID`) so nothing regresses.
- [x] Frontend: fetch the per-agent tools for EVERY agent's settings page (not just
      orchestrator) and render the Tools panel with the existing `renderToolControls`
      (read-only where per-agent toggling is not meaningful - a sub-agent's
      `request_input` is not operator-toggle-able; show it, do not offer a disable).
      Fix the "N tools" count wherever it reads the global list for a sub-agent.
- [x] Docs sync: CHANGELOG note the role-correct per-agent tools view.

## Definition of Done

- `GET /api/agents/{id}/tools` returns the role-scoped set: orchestrator -> full;
  a codex sub-agent -> exactly `["request_input"]`; a claude sub-agent -> `[]`.
  (test: `test_agent_tools_endpoint_is_role_scoped`)
- The sub-agent no longer shows the orchestrator's 18 tools anywhere in the UI.
  (manual: open a codex sub-agent - it shows 1 tool, not 18)
- Each agent's settings page renders its available-tools panel (like the
  orchestrator's). (manual: open a sub-agent's settings, see its tool(s))
- `ruff check .`, `mypy`, `python -m pytest` green; jsdom/frontend tests green.
  (cmd: `python -m pytest` and the web test suite)

## Notes

- Composes with the claude-MCP spike (20260723-193218): once a claude sub-agent
  gets `request_input`, a backend-aware `tools_for_role` reports it automatically.
- Umbrella 20260723-192825.
