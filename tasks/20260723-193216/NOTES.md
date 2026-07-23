# NOTES: reproducing the "sub-agent shows 18 tools" report

## What the trace found (two out-of-context frontend/backend passes)

- BACKEND (the latent bug): `GET /api/agent/tools` (`scufris/app.py`) returns the
  dashboard's GLOBAL, never-role-scoped `mcp` instance - all ~18 tools - with no
  agent/role parameter. Role scoping only happens in a SPAWNED per-turn server
  (`mcp_server.apply_role`, `_AGENT_ROLE_TOOLS={"request_input"}`), never at the
  endpoint. So any per-agent surface reading `/api/agent/tools` sees the
  orchestrator's full set.
- FRONTEND: no per-agent page currently renders a tools list. The "N tools" count
  (`agent-view.ts` -> `#agent-tools-link`) is ONLY on the LANDING orchestrator
  page; the orchestrator SETTINGS page renders the Tools card gated to
  `isOrchestrator`; a project agent's settings page renders NO tools section today.

## Conclusion

The reported "sub-agent shows 18 tools available" was the ORCHESTRATOR-scoped
count surfacing (the landing/orchestrator console is the only place tools render),
not a dedicated sub-agent panel - there was none. So this task is BOTH:

1. Fix the latent correctness gap: add a role- + backend-scoped
   `GET /api/agents/{id}/tools` (single source of truth via a new pure
   `mcp_server.role_tool_names`), so a sub-agent's real tool surface is queryable.
2. Deliver the transparency the user asked for: render a read-only Tools card on
   EVERY agent's settings page from that endpoint - a codex sub-agent shows its one
   tool (`request_input`), a mock/claude agent shows "none", the orchestrator keeps
   its writable operator console. Now the settings page never shows a sub-agent the
   orchestrator's eighteen.

Backend proof: `test_agent_tools_endpoint_is_role_scoped`. Frontend proof: three
new `agent-settings-view.test.ts` cases (renders the scoped panel / the "none"
note / not for the orchestrator).
