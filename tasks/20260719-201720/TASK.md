# Backend: surface agent tool-calls, token usage, model/tools

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,backlog,agent,llm

## Goal

Surface the agent's per-turn internals via the backend: capture the MCP tool
calls and token usage from the `codex exec --json` stream the agent already
parses, and expose the model/tools it uses.

## Notes

- Spike: tasks/20260719-180528/SPIKE.md (RECOMMENDED; probed data shapes there).
- `_run_codex_exec` (scufris/agent.py) already runs `--json` and parses
  `thread.started` for continuity. Extend that parse to also collect
  `item.completed`/`item.started` `mcp_tool_call` items (`server`, `tool`,
  `status`) and the `turn.completed` `usage`
  (`input_tokens`/`cached_input_tokens`/`output_tokens`/`reasoning_output_tokens`).
- Add `tool_calls: list[...]` and `usage: {...}` to `AgentReply`; `POST /api/chat`
  returns them with each reply (fill via `TurnOutcome` -> the agent -> AgentReply).
- Add `GET /api/agent/info` (model from settings, auth mode, agent_enabled) and
  `GET /api/agent/tools` (the Scufris MCP tools: name + description, from
  scufris/mcp_server.py's registry - NOT `codex mcp list`).
- Harness-first: fake the codex runner to emit sample `--json` lines and assert
  tool_calls + usage parse; endpoint tests via fakes.
- Model/tools are Scufris's own source of truth; context window is not exposed by
  the CLI (use input_tokens as the context-fill signal). Builds on tatr
  20260719-162356 / 162406 / 162419.
