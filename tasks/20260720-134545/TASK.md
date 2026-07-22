# Backend: run-one-tool endpoint + param schema for the 'try it' runner

- STATUS: OPEN
- PRIORITY: 21
- TAGS: feature,agent,backend,mcp

## Story

As a homelab operator debugging the agent, I want to run a single scufris MCP tool
by name with args and get its raw result back WITHOUT a chat turn, so I can answer
"does host_stats work right now?" in isolation. This task is the backend half: the
run-one-tool capability plus the per-tool parameter schema the UI form is generated
from. The interactive UI is task 20260722-213000 (built on this contract).

Consent model (decided with the user, see GOAL.md): the confirm step lives in the
UI; the backend refuses any tool in `disabled_tools`; there is NO new gating setting
(the tool set is already curated - fixed flags, bounded output, no arbitrary-command
tool; most are read-only, only `tatr_new` writes and it is bounded to tatr).

## Steps

- [ ] Add a `ToolParam` pydantic model (name: str, type: str, required: bool,
      description: str = "", default: object | None = None) in scufris/app.py and a
      `parameters: list[ToolParam] = []` field on `AgentTool` (keep the existing
      `args` names field - the tool card + health still use it).
- [ ] In `get_agent_tools` (app.py ~1324), populate `parameters` from each tool's
      `inputSchema`: iterate `properties`, map JSON-schema `type`
      (string/integer/number/boolean/...) to the param type, mark `required` from the
      schema's `required` list, carry `description`/`default` when present.
- [ ] Add `POST /api/agent/tools/{name}/run` taking body `{args: dict}`. Refuse a
      tool in `settings.disabled_tools` -> 403. Run it in-process via
      `mcp.call_tool(name, args)` (returns `tuple[list[ContentBlock], dict]`); build
      the response `{ok: true, text, structured}` where `text` is the concatenated
      `.text` of the TextContent blocks and `structured` is the dict block (may be
      empty). Do NOT go through codex/the agent.
- [ ] Map failures to controlled 4xx, never an uncontrolled 500: unknown tool -> 404
      (check the tool exists in `list_tools()` first, or match the "Unknown tool"
      ToolError), and `ToolError` from bad/missing/invalid args -> 422 with the error
      message in `detail`. Import `ToolError` from `mcp.server.fastmcp.exceptions`.
- [ ] Check the auth/path-prefix routing in app.py (~163-190): confirm
      `POST /api/agent/tools/{name}/run` is classified correctly (it is under the
      singular `/api/agent/` family, not `/api/agents`), and add it to the write/read
      classification wherever `/api/agent/tools` is handled so it is not mis-gated.
- [ ] Mirror the contract in the frontend types only (no UI here): extend the
      `AgentTool` interface in web/src/common.ts with `parameters: ToolParam[]` and
      add a `ToolParam` interface, so task 20260722-213000 can consume it.
- [ ] Tests in tests/ (FastAPI TestClient, harness-first): run host_stats -> 200 and
      `text` contains "hostname"; run tatr_ls -> 200; a tool in `disabled_tools` ->
      403; unknown tool -> 404; bad args (list_processes limit "notanint") -> 422;
      GET /api/agent/tools now returns `parameters` with types + required for a tool
      that has args (e.g. list_processes.limit integer, tatr_show.task_id required).

## Definition of Done

- `POST /api/agent/tools/{name}/run` runs one MCP tool in-process (bypassing the
  agent) and returns its result (test: `test_run_tool_host_stats_returns_result`).
- A disabled tool is refused 403, unknown tool 404, bad args 422 - never an
  uncontrolled 500 (test: `test_run_tool_rejects_disabled_unknown_and_badargs`).
- `GET /api/agent/tools` exposes per-tool `parameters` (name/type/required) distilled
  from `inputSchema` (test: `test_tools_endpoint_exposes_parameters`).
- Backend gate is green (cmd: `python -m pytest tests -q`) and types clean
  (cmd: `mypy scufris`).

## Notes

- Relevant files: scufris/app.py (`AgentTool` ~231, `get_agent_tools` ~1324,
  auth/path classification ~163-190), scufris/mcp_server.py (`mcp = FastMCP(...)`;
  `mcp.call_tool(name, args)` returns `tuple[list[ContentBlock], dict]`; output
  already capped at 20k, timeout 15s), web/src/common.ts (`AgentTool` ~190).
- Verified live: `mcp.call_tool('host_stats', {})` -> tuple(list[TextContent], dict);
  unknown tool / bad args / missing required arg all raise
  `mcp.server.fastmcp.exceptions.ToolError` (message distinguishes "Unknown tool:").
- Consent: confirm-only in the UI, refuse disabled_tools, NO new setting (GOAL.md).
- Depends on nothing new; the dependency 20260720-122517 (operator console) is CLOSED.
