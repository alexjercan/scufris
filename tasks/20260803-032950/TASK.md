# Make the health session count follow the orchestrator backend

- PRIORITY: 60
- TAGS: bug, v0.2.0, agents, backend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As an operator switching the orchestrator off Codex, I want the agent health
surface's session count to come from the orchestrator's own backend, so that a
claude or opencode orchestrator stops reporting a CODEX rollout count.

## Context

`scufris/health.py:258` calls `list_sessions(resolve_codex_home(settings), ...)`
unconditionally whenever `settings.agent_enabled`, so `session_count` and
`last_session` are codex readings regardless of the effective backend. Both the
legacy `/api/agent/health` and the scoped `/api/agents/{id}/health` carry it, so
the two agree and 20260801-100415's contract test passes - it is a residual
leak, not a regression.

Surfaced as a process signal in that task's round 1 review; explicitly out of
its scope (its DoD grep targets `scufris/app.py`, and `health.py` owns its own
probing).
