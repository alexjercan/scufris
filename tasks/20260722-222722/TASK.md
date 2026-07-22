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
