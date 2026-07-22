# T1: orchestrator-only scufris MCP scoping (thread is_orchestrator; register server only for orchestrator)

- STATUS: OPEN
- PRIORITY: 36
- TAGS: spike,telegram,agent,mcp,backend

## Goal

Make the scufris MCP server ORCHESTRATOR-ONLY. Thread an `is_orchestrator`
flag from the run call site (`app.py:_launch_agent_turn` knows
`agent.id == ORCHESTRATOR_ID`) through `backend.stream` ->
`_stream_app_server` -> `_mcp_overrides`, and register the scufris MCP server
ONLY for the orchestrator. Regular agents stop receiving it and draw their
tools from their own project `.config` / `.skills` (this is a deliberate
behavior change - today every agent gets the 8 tools).

This is the foundation the Telegram frontend builds on: control tools may only
exist somewhere the orchestrator alone can reach.

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q3, and Context "no orchestrator
  scoping today").
- Foundation task - T2 (control tools) and T3 (prune) build on it.
- Grounding: `scufris/agent.py:_mcp_overrides`/`_server_override`;
  `backends.py` `CodexBackend.stream` / `_stream_app_server` signatures;
  `app.py:_launch_agent_turn` (~1099) passes global `settings` to every
  `backend.stream`; `agent_store.ORCHESTRATOR_ID`.
- Claude backend has no MCP wiring today - keep the claude/mock paths coherent
  (the flag is a no-op there until T2's claude follow-up).
- Test: a regular-agent turn registers NO scufris server; an orchestrator turn
  does.
- spike-seeded; plan into steps with /plan before /work.
