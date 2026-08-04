# Default to app_server backend; add offline mock agent

- PRIORITY: 50
- TAGS: feature, agent, config
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Stop defaulting to exec: enable the agent and select the app_server (streaming)
backend out of the box, and add a mock backend so the streaming UI can be
developed/demoed with no codex login or network.

## Implementation

- `scufris/config.py`: `agent_enabled` default `True`; `agent_backend` default
  `"app_server"`; added `"mock"` to the backend literal.
- `scufris/agent.py`: `MockAgent` - a canned in-process agent (no subprocess or
  network) that streams a reasoning ("thinking") section, a tool-call chip, and
  token-by-token markdown (incl. a code block) with small delays, and fakes
  session switch/new/reset with an in-memory id. `build_agent` returns it when
  `agent_backend == "mock"`.
- `.env.example`: documents the new defaults and the mock backend.

## Tests

- `test_config.py`: defaults are enabled + app_server; `SCUFRIS_AGENT_BACKEND`
  parses from env.
- `test_agent.py`: `build_agent` selects the mock; `MockAgent.chat_stream`
  emits reasoning + tool + text deltas + done and drives the fake session.
- `test_app.py`: the `*_503_when_disabled` tests now disable the agent explicitly
  (they no longer rely on the old default-off), and the config endpoint test pins
  the flag it echoes.

## Definition of Done

- [x] `scufris serve` (no env) runs the streaming agent by default.
- [x] `SCUFRIS_AGENT_BACKEND=mock` streams the full UI offline (verified through
      the real SSE endpoint: reasoning x3, tool x1, text x33, done).
- [x] Disabled-agent behavior still reachable via `SCUFRIS_AGENT_ENABLED=0`.
- [x] 122 pytest + 58 frontend green.
