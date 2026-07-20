# A4: Agents dashboard page (live status list; fold Projects into agent creation)

- STATUS: OPEN
- PRIORITY: 22
- TAGS: spike,agents,frontend

## Goal

The Agents dashboard page: the LIST view polls `GET /api/agents` for coarse
live status (state, last activity, tokens), reusing the Stats page polling +
client-side sparkline patterns and the pure-render + injected-actions seam. The
FOCUSED/open agent view uses SSE (`GET /api/agents/{id}/events`) relayed from the
supervisor event bus (ADR-001) for live token streaming - drop-safe, replays on
reconnect. Fold the standalone Projects page into the agent-creation flow
(project becomes a picker). This is what turns the AGENT/STATS gimmicks into a
real cockpit.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 5).
- Depends on: 20260720-221929 (A1), 20260720-221942 (A3).
- Stepless direction-level task: run /plan before /work.
