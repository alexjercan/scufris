# B5b: retire the Agent protocol - route the orchestrator through get_backend (unified backend path)

- STATUS: OPEN
- PRIORITY: 33
- TAGS: agents,backend

## Goal

Make the reserved orchestrator (from B5a) run its chat through the SAME path as
project agents: `get_backend(agent.backend).stream(...)` via `_launch_agent_turn`
+ the supervisor + event bus, instead of the `CodexCliAgent`/`Agent`-protocol
direct path. This removes the second abstraction. After B5b the orchestrator's
turns are supervised runs like any agent (still single active session; B5c adds
multi-session). The `Agent` protocol / `CodexCliAgent` / `AgentHandle` shrink or
retire once nothing uses them.

## Coarse steps (/plan expands)

- [ ] Point the landing chat endpoints (`/api/chat`, `/api/chat/stream`) at the
      orchestrator agent record + `_launch_agent_turn` (or redirect the landing
      page to `/api/agents/orchestrator/chat`). Keep the SSE frame shape.
- [ ] Carry the orchestrator's active session_id in the agent model (settings or
      a dedicated field) so a turn resumes it; capture StreamDone.session_id.
- [ ] Retire `CodexCliAgent`, `AgentHandle`, `build_agent`, the `Agent` protocol
      and the Agent-shaped Mock/Disabled impls once the endpoints no longer use
      them (the disabled-state gate moves to the backend/settings). Update every
      test that mocks `Agent`.
- [ ] Preserve `/api/agent/info`/`config`/`health`/`context`/`usage`/`account`
      by sourcing them from the orchestrator record + backend read_status, or
      re-point them.

## Definition of Done

- The orchestrator chat runs through `get_backend(...).stream()` via the
  supervisor (test: a landing/orchestrator turn is a supervised run).
- The `Agent` protocol + CodexCliAgent are gone (or reduced to nothing the app
  imports) (cmd: `grep -rn "class CodexCliAgent" scufris/`).
- Full suite green.
- manual: the landing/orchestrator chat still works end to end.

## Notes
- Depends on: B5a (20260721-112439). Blocks: B5c.
- HIGHEST-RISK slice (retires an abstraction + rewrites the landing chat path +
  many test doubles). Probe the exec/app_server session semantics before wiring.
- Retiring `_stream_codex_exec`/`_run_codex_exec` (the last exec users) may land
  here or in B5e - whichever removes the final reference cleanly.
