# T2: orchestrator control MCP tools over the local HTTP API (list/create project, create/run/message agent)

- STATUS: OPEN
- PRIORITY: 35
- TAGS: spike,telegram,agent,mcp,backend

## Goal

Add curated CONTROL tools to the (orchestrator-only) scufris MCP server so the
orchestrator can DO dashboard actions, not just observe: `list_projects`,
`create_project`, `create_agent`, `run_agent`, `message_agent` (steer). They
call the dashboard's own HTTP API at `127.0.0.1:<port>` via httpx - crossing
the MCP-subprocess boundary cleanly and reusing each endpoint's validation and
lifecycle (the MCP process cannot touch the live in-app Supervisor). Keep the
existing `_run` contract: fixed shapes, timeout, bounded output.

## Steps

- [ ] Add a bounded local-API helper in `scufris/mcp_server.py` (sibling to
      `_run`): an httpx call to `http://<host>:<port><path>` with a timeout and
      `_MAX_OUTPUT` truncation, returning text (never raising - errors come back
      as `error: ...` like `_run`). Read base URL from env
      (`SCUFRIS_API_BASE`, e.g. `http://127.0.0.1:8000`).
- [ ] Set that env when the orchestrator server is spawned: in
      `scufris/agent.py` `_mcp_overrides` (orchestrator branch from T1), add
      `SCUFRIS_API_BASE` (from `settings.host`/`settings.port`) to the scufris
      server's env alongside the existing `SCUFRIS_DISABLED_TOOLS`.
- [ ] Implement `list_projects()` -> `GET /api/projects` (compact table).
- [ ] Implement `create_project(...)` -> `POST /api/projects/new` (or
      `/api/projects`); read the request models in `app.py` first and mirror the
      required fields exactly. Validate/normalize inputs like `tatr_new` does.
- [ ] Implement `create_agent(...)` -> `POST /api/agents`; mirror the create
      request model's fields (project, backend, name, permission_mode, ...).
- [ ] Implement `run_agent(agent_id, ...)` -> `POST /api/agents/{id}/run` and
      `message_agent(agent_id, text)` -> `POST /api/agents/{id}/chat`; mirror the
      request models and return a bounded status line.
- [ ] Register all five as `@mcp.tool()` with model-facing docstrings; confirm
      they are only reachable via the orchestrator-only server from T1.
- [ ] Tests (`tests/test_mcp_server.py`): drive each tool against a
      `respx`-stubbed local API (or a FastAPI `TestClient` base URL), asserting
      the right method+path+body and bounded text; include an error-path case
      (non-2xx -> `error:` text). Add the five names to the registration-set
      assertion.

## Definition of Done

- The five control tools exist on the scufris server and call the local API with
  the correct method/path/body, returning bounded text.
  (test: `` `test_control_tools_call_local_api` ``)
- A non-2xx response yields an `error:` string, never an exception.
  (test: `` `test_control_tool_error_path` ``)
- `nix flake check` is green. (cmd: `nix flake check`)

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
