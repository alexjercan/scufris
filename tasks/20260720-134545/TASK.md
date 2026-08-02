# Backend: run-one-tool endpoint + param schema for the 'try it' runner

- PRIORITY: 21
- TAGS: feature, agent, backend, mcp
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Add a `ToolParam` pydantic model (name: str, type: str, required: bool,
      description: str = "", default: object | None = None) in scufris/app.py and a
      `parameters: list[ToolParam] = []` field on `AgentTool` (keep the existing
      `args` names field - the tool card + health still use it).
- [x] In `get_agent_tools` (app.py ~1324), populate `parameters` from each tool's
      `inputSchema`: iterate `properties`, map JSON-schema `type`
      (string/integer/number/boolean/...) to the param type, mark `required` from the
      schema's `required` list, carry `description`/`default` when present.
- [x] Add `POST /api/agent/tools/{name}/run` taking body `{args: dict}`. Refuse a
      tool in `settings.disabled_tools` -> 403. Run it in-process via
      `mcp.call_tool(name, args)` (returns `tuple[list[ContentBlock], dict]`); build
      the response `{ok: true, text, structured}` where `text` is the concatenated
      `.text` of the TextContent blocks and `structured` is the dict block (may be
      empty). Do NOT go through codex/the agent.
- [x] Map failures to controlled 4xx, never an uncontrolled 500: unknown tool -> 404
      (check the tool exists in `list_tools()` first, or match the "Unknown tool"
      ToolError), and `ToolError` from bad/missing/invalid args -> 422 with the error
      message in `detail`. Import `ToolError` from `mcp.server.fastmcp.exceptions`.
- [x] Check the auth/path-prefix routing in app.py (~163-190): confirm
      `POST /api/agent/tools/{name}/run` is classified correctly (it is under the
      singular `/api/agent/` family, not `/api/agents`), and add it to the write/read
      classification wherever `/api/agent/tools` is handled so it is not mis-gated.
- [x] Mirror the contract in the frontend types only (no UI here): extend the
      `AgentTool` interface in web/src/common.ts with `parameters: ToolParam[]` and
      add a `ToolParam` interface, so task 20260722-213000 can consume it.
- [x] Tests in tests/ (FastAPI TestClient, harness-first): run host_stats -> 200 and
      `text` contains "hostname"; a tool in `disabled_tools` -> 403; unknown tool ->
      404; bad args (list_processes limit "notanint") -> 422; GET /api/agent/tools now
      returns `parameters` with types + required for a tool that has args
      (list_processes.limit integer, tatr_show.task_id required). Used host_stats +
      list_processes for the run tests (NOT tatr_ls) so they stay green in the
      `nix flake check` sandbox, which has no tatr on PATH.

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

## Closing record

What changed:
- `scufris/app.py`: new `ToolParam` model + a `parameters: list[ToolParam]` field on
  `AgentTool`; `_tool_parameters(input_schema)` distills `properties` + top-level
  `required` into typed params (unknown/missing type -> "string"; malformed schema ->
  []). `get_agent_tools` now populates `parameters` alongside the existing `args`.
- New route `POST /api/agent/tools/{name}/run` with `ToolRunRequest{args}` ->
  `ToolRunResult{ok,text,structured}`. Runs the tool in-process via
  `mcp.call_tool`, bypassing codex. Refuses `disabled_tools` (403), unknown tool
  (404, checked against `list_tools()`), and `ToolError` from FastMCP arg validation
  (422). No gating setting.
- `web/src/common.ts`: mirrored `ToolParam` + `AgentTool.parameters` (types only, no
  UI); updated the three TS test fixtures that construct `AgentTool` to add
  `parameters: []` so the frontend keeps compiling.
- Tests: `test_tools_endpoint_exposes_parameters`,
  `test_run_tool_host_stats_returns_result`,
  `test_run_tool_rejects_disabled_unknown_and_badargs` in tests/test_app.py.

Decisions / difficulties:
- The "auth/path classification" step turned out to be a non-change: app.py ~160-189
  is `_route_tags` (OpenAPI tags only), NOT auth gating. The singular `/api/agent/`
  prefix already classifies the run route as "settings"; the plural `/api/agents`
  check precedes it and does not catch it. There is no global write-auth gate
  (`settings_writable` guards only the settings store, which the runner does not
  touch), so nothing needed adding. Ticked as verified.
- `FastMCP.call_tool` is annotated `-> Sequence[ContentBlock] | dict` but at runtime
  returns the 2-tuple `(content_blocks, structured_dict)` (verified live). Unpacked
  it defensively via `cast(Any, ...)` + a tuple/len check so a future shape change
  degrades to `structured={}` instead of 500-ing.
- Kept `tatr_ls` out of the run tests: it shells out to the tatr CLI, which is absent
  in the `nix flake check` sandbox (conftest skips `needs_tatr`). host_stats +
  list_processes exercise the same code paths with no external binary.

Verification (in the dev shell, worktree): `ruff check` clean, `ruff format` clean,
`mypy scufris` clean (21 files), full `python -m pytest` green, and the frontend
`prettier --check` / `npm run lint` / `npm run test` (155 tests) all green after the
type-mirror change.

Self-reflection: the plan's "add it to the write/read classification" step baked in
an assumption (that a write-auth gate existed) that the code did not bear out -
reading `_route_tags` first would have phrased it as a verify-first step. Net cheap
because I read the routing before touching it. The type-mirror-breaks-fixtures risk
was anticipated from the lessons ledger and handled in the same pass rather than
discovered by a red frontend build.
