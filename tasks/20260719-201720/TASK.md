# Backend: surface agent tool-calls, token usage, model/tools

- STATUS: CLOSED
- PRIORITY: 12
- TAGS: feature, backlog, agent, llm
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Surface the agent's per-turn internals via the backend: capture the MCP tool
calls and token usage from the `codex exec --json` stream the agent already
parses, and expose the model/tools it uses.

## Steps

- [x] `scufris/agent.py` models: `ToolCall` (server, tool, status), `TokenUsage`
      (input/cached_input/output/reasoning tokens). Extend `AgentReply` with
      `tool_calls: list[ToolCall] = []` and `usage: TokenUsage | None = None`, and
      `TurnOutcome` with the same two fields (default empty).
- [x] Replace `_parse_thread_id` with `_parse_events(stdout) -> (thread_id,
      tool_calls, usage)`: thread id from `thread.started`; a `ToolCall` per
      `item.completed` `mcp_tool_call` (server/tool/status); `usage` from
      `turn.completed`. Wire into `_run_codex_exec`; map onto `AgentReply` in
      `CodexCliAgent.chat`.
- [x] `scufris/app.py`: `GET /api/agent/info` (model, auth_mode, agent_enabled)
      and `GET /api/agent/tools` (Scufris MCP tools via `mcp.list_tools()`: name +
      description). `POST /api/chat` already returns `AgentReply`, so tool_calls +
      usage ride along once the model has them.
- [x] Tests: `_parse_events` unit (sample JSON lines -> tool_calls + usage); a
      fake-codex integration test (script emits the events + writes `-o`);
      `CodexCliAgent.chat` carries tool_calls/usage; `/api/agent/info`,
      `/api/agent/tools`, and `/api/chat` metadata via fakes.
- [x] LIVE VERIFY on this host: a `/api/chat` turn that uses a tool returns
      `tool_calls` (scufris/...) + `usage` (real tokens); `/api/agent/tools` lists
      the 3 tools; `/api/agent/info` shows the model. Record evidence.
- [x] `ruff`/`mypy`/`pytest` green.

## Definition of Done

- `POST /api/chat` replies carry `tool_calls` (server/tool/status) and `usage`
  (token counts) parsed from the `--json` stream; `GET /api/agent/info` and
  `GET /api/agent/tools` return the model/auth/enabled and the tool registry.
- Live-verified on this host; tests green (parse unit + fake-codex integration +
  endpoints).

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

## Implementation

- `scufris/agent.py`: new `ToolCall` (server/tool/status) and `TokenUsage`
  (input/cached_input/output/reasoning) models; `AgentReply` + `TurnOutcome` gain
  `tool_calls` and `usage`. `_parse_thread_id` -> `_parse_events(stdout)` returns
  (thread_id, tool_calls, usage): thread id from `thread.started`, a `ToolCall`
  per `item.completed` `mcp_tool_call`, usage from `turn.completed`. Wired into
  `_run_codex_exec` and mapped onto `AgentReply` in `CodexCliAgent.chat`. The
  agent already ran `--json`, so this is pure extra parse - no new codex work.
- `scufris/app.py`: `GET /api/agent/info` (model, auth_mode, enabled from
  settings) and `GET /api/agent/tools` (the Scufris MCP tools via
  `mcp.list_tools()` - name + description). `POST /api/chat` returns `AgentReply`,
  so tool_calls + usage ride along.
- Tests: `_parse_events` unit (tools + usage from sample JSON, malformed line
  skipped); a fake-codex integration test emitting the events; `chat` carries the
  metadata; `/api/agent/info`, `/api/agent/tools` (real MCP list), and `/api/chat`
  metadata. ruff+mypy+pytest green.

### Live verification (DoD)

On this host: `/api/agent/info` -> `{model: gpt-5.5, auth_mode: chatgpt, enabled:
true}`; `/api/agent/tools` -> `[host_stats, tatr_ls, tatr_show]`; a `/api/chat`
turn using a tool returned `tool_calls: [{scufris/host_stats, completed}]` and
real `usage` (input 47061, cached 34944, output 87, reasoning 32).
