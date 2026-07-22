# T2: orchestrator control MCP tools over the local HTTP API (list/create project, create/run/message agent)

- STATUS: CLOSED
- PRIORITY: 35
- TAGS: spike, telegram, agent, mcp, backend

## Goal

Add curated CONTROL tools to the (orchestrator-only) scufris MCP server so the
orchestrator can DO dashboard actions, not just observe: `list_projects`,
`create_project`, `create_agent`, `run_agent`, `message_agent` (steer). They
call the dashboard's own HTTP API at `127.0.0.1:<port>` via httpx - crossing
the MCP-subprocess boundary cleanly and reusing each endpoint's validation and
lifecycle (the MCP process cannot touch the live in-app Supervisor). Keep the
existing `_run` contract: fixed shapes, timeout, bounded output.

## Steps

- [x] Added a bounded local-API helper in `scufris/mcp_server.py` (`_api_call`,
      sibling to `_run`): httpx to `SCUFRIS_API_BASE + path` with a timeout and
      `_MAX_OUTPUT` truncation, returning text (never raising - non-2xx and network
      errors come back as `error: ...`). `_api_base()` reads `SCUFRIS_API_BASE`
      (default `http://127.0.0.1:8000`).
- [x] Set that env when the orchestrator server is spawned: `scufris/agent.py`
      `_mcp_overrides` orchestrator branch now injects `SCUFRIS_API_BASE`
      (`http://{settings.host}:{settings.port}`) into the scufris server's env
      (alongside `SCUFRIS_DISABLED_TOOLS` when set).
- [x] `list_projects()` -> `GET /api/projects`, rendered as a compact table.
- [x] `create_project(name, cwd, language, description)` -> `POST /api/projects`
      (register an EXISTING dir; mirrors `ProjectCreate`). Chose `/api/projects`
      over `/api/projects/new` for v1 - `/new` needs a `base` from the configured
      base dirs, a poorer fit for a chat tool; registering a path is the common case.
- [x] `create_agent(name, project_id, backend?, model?, description, goal,
      permission_mode)` -> `POST /api/agents` (mirrors `AgentCreate`; omits
      empty backend/model so the server stamps defaults).
- [x] `run_agent(agent_id, goal?)` -> `POST /api/agents/{id}/run` (returns the run
      state); `message_agent(agent_id, message)` -> `POST /api/agents/{id}/chat`,
      which streams SSE - the tool collects the assistant reply from the frames
      (longer `_CHAT_TIMEOUT` since it runs a full turn).
- [x] Registered all five as `@mcp.tool()` with model-facing docstrings; they live
      on the orchestrator-only server from T1, so a regular agent can't reach them.
- [x] Tests (`tests/test_mcp_server.py`): each tool driven against a `respx`-stubbed
      local API asserting method+path+body and bounded text, an SSE-collect test, a
      stream-error test, a non-2xx error-path test and a network-error test; the five
      names added to the registration-set assertion. Plus a `test_agent.py` test that
      the orchestrator server env carries `SCUFRIS_API_BASE`.

## Definition of Done

- The five control tools exist on the scufris server and call the local API with
  the correct method/path/body, returning bounded text.
  (test: `` `test_create_project_posts_body_and_returns_result` ``,
  `` `test_create_agent_posts_body_and_omits_empty_backend` ``,
  `` `test_run_agent_posts_goal_and_returns_state` ``,
  `` `test_list_projects_calls_api_and_formats` ``,
  `` `test_message_agent_collects_sse_reply` ``)
- A non-2xx response yields an `error:` string, never an exception.
  (test: `` `test_control_tool_error_path` ``)
- The orchestrator server env carries the API base.
  (test: `` `test_mcp_overrides_injects_api_base_for_orchestrator` ``)
- `nix flake check` is green EXCEPT the pre-existing mypy red (task 20260720-174021);
  ruff + pytest legs pass and the changed source files add zero mypy errors.
  (cmd: `nix flake check`)

## Notes

- Spike: tasks/20260722-221359/SPIKE.md (Q2).
- Depends on: T1 (orchestrator-only scoping).
- Endpoints to wrap: `GET/POST /api/projects`, `POST /api/projects/new`,
  `POST /api/agents`, `POST /api/agents/{id}/run`, `POST /api/agents/{id}/chat`.
  Base URL from settings (`host`/`port`) passed to the MCP server via env.
- Codex-first: the claude backend has no MCP wiring - either fold a claude
  `--mcp-config` step in here or split it as a follow-up (see SPIKE open
  questions).
- Test: each tool against a stubbed/real local API (respx or a FastAPI
  TestClient), asserting bounded text and correct endpoint calls.
- spike-seeded; plan into steps with /plan before /work.

## Implementation (close)

Added five orchestrator-only control tools on the scufris MCP server that call the
dashboard's own HTTP API via a bounded `_api_call` helper (`SCUFRIS_API_BASE`,
injected by `agent._mcp_overrides` for the orchestrator server). The tools mirror
the endpoints' request models (`ProjectCreate`, `AgentCreate`, `AgentRunRequest`,
`AgentChatRequest`), normalize/guard inputs before any HTTP call, and return bounded
text - non-2xx and network failures become `error: ...` strings, never exceptions.

Design choices worth the why:
- HTTP call-back, not in-process store access: the MCP server is a SEPARATE process
  and cannot touch the live in-app supervisor (run/steer needs it), so control tools
  cross back over the local API - reusing every endpoint's validation. (SPIKE Q2.)
- `create_project` targets `POST /api/projects` (register an existing directory), not
  `/api/projects/new` (which needs a `base` from the configured base dirs) - the
  register-a-path case is the better fit for a chat-driven tool.
- `message_agent` consumes the chat endpoint's SSE frames and extracts the assistant
  reply; it uses a longer `_CHAT_TIMEOUT` (120s) since it runs a full agent turn.
  Tradeoff: an MCP tool call can block for the turn's duration - acceptable for v1
  (a steer is inherently synchronous); noted for the reviewer.

Codex-first (SPIKE open question): these tools only reach the orchestrator when it
runs on codex - the claude backend still has no MCP wiring. Left as a follow-up; not
in scope for this task.

Difficulties: none major. The SSE frame format had to be read from `_relay_bus_sse`
(`id:`/`data:` lines, done frame carries `reply.text`) so the parser and its test
match the real endpoint, not a guess.

Verification: ruff + full pytest green (354 tests, incl. the new control-tool tests);
changed source files (`mcp_server.py`, `agent.py`) add zero mypy errors. `nix flake
check` mypy leg remains pre-existing-red (task 20260720-174021).

Self-reflection: applied the T1 retro lesson - this task did not change a Protocol
signature, so no test-double ripple; but I still grepped for tool-registration
assertions up front and updated the set in the same pass rather than discovering it
by a failing test.
