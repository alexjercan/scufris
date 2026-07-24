# Thread parent_agent_id into child launch context for request_input escalation

- STATUS: OPEN
- PRIORITY: 38
- TAGS: spike, agents, sessions

## Goal

Make "which agent/session spawned me" EXPLICIT. Inject `parent_agent_id` into a
sub-agent's launch context alongside the existing `SCUFRIS_AGENT_ID`
(`agent.py`/`mcp_server.py`), record it on the child's session in the index
(part 1), and keep the `request_input` -> `pending_agents` -> `message_agent`
loop as the escalation channel. This replaces today's implicit "the
orchestrator is everyone's parent" with a real link, so escalation and the
session switcher can attribute a child to its actual spawner.

## Notes

- Spike: tasks/20260724-111839/SPIKE.md (part 3)
- Depends on part 1's index carrying `parent_agent_id`.
- No CLI offers native child->parent callback; the escalation stays on scufris's
  own channel (matches LangGraph interrupt/resume, AutoGen user-proxy). MCP
  elicitation is the spec-level fit if a standard is wanted later.

