# T3: prune the MCP surface (drop tatr_* tools; keep host tools orchestrator-scoped; update steering/tests)

- STATUS: OPEN
- PRIORITY: 34
- TAGS: spike,telegram,agent,mcp

## Goal

Prune the MCP tool surface for the orchestrator-only world. DROP `tatr_ls`,
`tatr_show`, `tatr_new` - `tatr` is a skill the orchestrator runs via `Bash`,
so a dedicated MCP wrapper is redundant once the server is orchestrator-scoped.
KEEP `host_stats`, `disk_usage`, `list_processes`, `list_agents`,
`agent_status` (now orchestrator-only via T1, joined by T2's control tools).
Update the tool-steering preamble and any docs/tests that name the removed
tools.

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q4).
- Depends on: T1.
- Caveat: `tatr_new` wrote from OUTSIDE the model's read-only sandbox; via Bash
  the orchestrator needs a write-capable permission mode to create tasks (SPIKE
  open question - confirm the orchestrator's default mode).
- Update `tests/test_mcp_server.py` tool-set assertions and
  `agent.py` STEERING_PREAMBLE references to tatr tools.
- spike-seeded; plan into steps with /plan before /work.
