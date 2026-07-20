# A5: orchestrator observation MCP tools (list_agents, agent_status)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: spike,agents

## Goal

Orchestrator observation (read-only, v1): give the main chat agent MCP tools
`list_agents` and `agent_status(id)` (built on the A2 status contract) so I can
ask the orchestrator "what is agent-N working on" and it answers by reading that
agent's status. No steering in v1 - observe + report only.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 4; steering deferred).
- Depends on: 20260720-221935 (A2).
- Stepless direction-level task: run /plan before /work.
