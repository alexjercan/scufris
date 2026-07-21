# B4: per-agent chat endpoint (message -> stream, resume the agent session) + transcript

- STATUS: OPEN
- PRIORITY: 38
- TAGS: agents,backend


## Goal

A per-agent multi-turn CHAT endpoint: `POST /api/agents/{id}/chat` streams a turn
via `get_backend(agent.backend).stream(prompt=message, session_id=
agent.session_id, cwd=project.cwd, permission_mode=agent.permission_mode)` through
the SAME supervisor + event bus as `run`, persisting the session id (one session
per agent, resumed each turn). `GET /api/agents/{id}/transcript` returns that
session's history so the UI can rebuild the conversation.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation B4). The backends already resume by session_id.
- Depends on: 20260721-112430 (B2), 20260721-112432 (B3).
