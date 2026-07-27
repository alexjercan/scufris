# Split MCP into scufris + den + sub-agent servers; per-server live health in settings

- STATUS: CLOSED
- PRIORITY: 58
- TAGS: feature,mcp,agents,frontend,backend,den

## Flow State

- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Outcome

Implemented across all four phases; as-built record + difficulties in
`NOTES.md`, rationale in `DECISION.md`. Three single-audience MCP servers
(scufris/den/agent) with a physical audience split, a live in-process per-server
health probe, and a grouped "MCP tools" settings section with per-server dots
and per-tool bulbs. Gate: `nix flake check` (ruff+mypy+pytest) green; web
`npm run ci` green (190 tests + build). DoD proofs all pass, including the new
`test_servers_expose_disjoint_tool_sets`, the registration tests
(`test_orchestrator_registers_scufris_and_den`,
`test_subagent_registers_only_callback_server`), and the health tests
(`test_mcp_health_den_warn_when_unconfigured`, `test_mcp_health_marks_disabled`).

## Story

As the operator, I want the one role-scoped scufris MCP split into three
single-audience servers - `scufris` (orchestrator agentic), `den` (my journal +
macros life tools), and a sub-agent callback server (request_input +
report_back) - and I want the settings page to show an "MCP tools" section: a
summary light (green all healthy / amber partial / red failing) that expands
into a per-server view, each server with its own status dot and its tools as
cards each carrying a green/red bulb. So that the three concerns are physically
separated (a sub-agent can never even register the orchestrator/den servers)
and I can see at a glance which servers and tools are live.

See `DECISION.md` for the mechanism (3 modules, physical audience split) and
the health semantics (in-process live probe). See `NOTES.md` after
implementation for the as-built record.

## Steps

### Phase A - Backend split into three servers

- [x] Create `scufris/mcp_common.py` holding the lean, dependency-light shared
      helpers moved out of `mcp_server.py`: `_run` (the `shutil.which` +
      `subprocess.run` wrapper), `_MAX_OUTPUT`, `_TIMEOUT_SECONDS`, and the
      HTTP `_api_call` + its `SCUFRIS_API_BASE` reader. No psutil / agent_store
      / FastMCP imports here, so `den_mcp_server` can import it without dragging
      in the orchestrator's deps.
- [x] Create `scufris/den_mcp_server.py`: `mcp = FastMCP("den")`, move the 9
      `journal_*` and 3 `macros_*` tools plus `_den_path` / `_journal` from
      `mcp_server.py`; import `_run` etc. from `mcp_common`. Add a `main()` that
      configures logging, runs `apply_disabled_tools(_disabled_tools())`, and
      `mcp.run()`; wire `python -m scufris.den_mcp_server` (an `if __name__`
      block).
- [x] Create `scufris/agent_mcp_server.py`: `mcp = FastMCP("agent")` (a DISTINCT
      id - see DECISION), move `request_input` + `report_back` and their helpers
      (`_self_agent_id`, the `SCUFRIS_ORCH_SESSION_ID` reader if used), importing
      `_api_call` from `mcp_common`. Add a `main()` + `python -m` entry. No
      disabled-tools filtering (the callbacks are not operator-hidable).
- [x] Trim `scufris/mcp_server.py` to the 17 orchestrator agentic tools
      (host_stats, disk_usage, list_processes, list_agents, agent_status, the 5
      project CRUD, the 5 agent control, pending_agents, acknowledge). Remove
      the moved tools and helpers; import shared helpers from `mcp_common`.
      Retire `apply_role`, `role_tool_names`, `ROLE_ORCHESTRATOR`, `ROLE_AGENT`,
      `_AGENT_ROLE_TOOLS`, `_role()` - the physical split replaces them. Keep
      `apply_disabled_tools` / `_disabled_tools` and `main()`.
- [x] Refactor `agent.scufris_mcp_server()` to return a LIST of server specs for
      the turn (each carrying its own `server_id`, command, args, env) instead
      of a single `ScufrisMcpServer`. Orchestrator turn -> `scufris`
      (mcp_server) + `den` (den_mcp_server, ONLY when `settings.den_path` is
      set, carrying `SCUFRIS_DEN_PATH`); sub-agent turn (`agent_id` set) ->
      the callback server (agent_mcp_server, id `scufris`, carrying API base +
      `SCUFRIS_AGENT_ID`). `SCUFRIS_DISABLED_TOOLS` rides the two orchestrator
      servers only. Returns `[]` when `agent_tools_enabled` is false or a
      sub-agent turn has no id. The callback server registers under id `agent`.
- [x] Update `agent._mcp_overrides` (codex) to iterate the returned list and
      `_server_override(spec.server_id, ...)` each, dropping the single-server
      assumption. Keep the `settings.mcp_servers` and `approval_policy="never"`
      tail. Guard the built-in ids (`scufris`, `den`) against operator
      `mcp_servers` collisions in the existing `spec.id == "scufris"` check.
- [x] Update `backends._scufris_claude_args` to build a `mcpServers` dict with
      an entry per returned spec and an `--allowedTools` wildcard per id
      (orchestrator turn: `mcp__scufris__*` + `mcp__den__*` when den present;
      sub-agent turn: `mcp__agent__*`), keeping `--strict-mcp-config`.
- [x] Update `app.py` tool endpoints for multiple servers:
      `_as_agent_tool` takes the real `server` id (stop hardcoding `"scufris"`);
      `GET /api/agent/tools` aggregates the orchestrator's servers (import
      `scufris.mcp_server` + `scufris.den_mcp_server`, tag each tool with its
      id, apply `disabled_tools`); `GET /api/agents/{id}/tools` picks the module
      set by audience (orchestrator id -> scufris+den; other -> agent server);
      `run_agent_tool` resolves the named tool across the correct module set.
      `_ensure_den_path` stays before running a den tool in-process.

### Phase B - Per-server live health probe

- [x] Add `scufris/mcp_health.py` (or a function in `health.py`) that, given a
      list of `(server_id, module)` for an audience, probes each IN-PROCESS:
      import the module's `mcp`, `await mcp.list_tools()`, and run readiness
      checks - `den`: `den_path` set AND `shutil.which("today")` and
      `shutil.which("macros")` present; `scufris`/agent: tool count > 0. Return
      a per-server record: `status` ok|warn|error, a short `detail`, and the
      tool list each with `enabled` (respecting `disabled_tools`) and an
      `available` flag (false when its server is unhealthy). Status rules:
      error if list_tools raised or count 0; warn if degraded (den_path unset /
      CLI missing, or some tools disabled); else ok.
- [x] Add response models (`McpServerHealth { id, status, detail, tools:
      list[AgentTool] }`, reusing `AgentTool` with its `enabled` field) and
      endpoints: `GET /api/agent/mcp` (orchestrator: scufris + den) and
      `GET /api/agents/{id}/mcp` (that agent's audience servers). 404 unknown
      agent; empty list when the agent's backend wires no scufris MCP
      (opencode/mock), matching `_agent_has_scufris_mcp`.

### Phase C - Settings UI grouped health view

- [x] Add TS types in `web/src/common.ts`: `McpServerHealth` (id, status:
      "ok"|"warn"|"error", detail, tools: AgentTool[]) and a status enum reuse.
- [x] In `web/src/settings-view.ts`, add `renderMcpServers(servers,
      actions)`: a section titled "MCP tools" with a SUMMARY dot (aggregate:
      red if any server error, amber if any warn, else green) and one
      collapsible `<details>` per server. Each `<summary>` shows the server id,
      tool count and a `health__dot--{status}`; the body is a `tool-grid` of
      `toolCard`s, each with a green/red bulb (reuse `health__dot` classes:
      green when enabled+available, red otherwise) added to `tool-card__head`.
      Keep the operator toggle (drives `disabled_tools`) and the "try it"
      runner for the orchestrator's enabled tools - i.e. this REPLACES the flat
      `renderToolControls` grid with the grouped, health-aware version.
- [x] Wire the data: in `web/src/agent-settings-view.ts` fetch
      `/api/agent/mcp` (orchestrator) or `/api/agents/{id}/mcp` (agent) in the
      existing parallel `Promise.all`, and render the new section where the
      Tools grid is today (orchestrator globals block; and the per-agent tools
      panel for sub-agents).
- [x] Add minimal CSS in `web/src/style.css` for the collapsible server rows
      and the per-card bulb placement, reusing the existing `--green/--amber/
      --red` vars and `health__dot` sizing.

### Phase D - Tests, examples, docs

- [x] Backend tests (`tests/`): den tools serve from `den_mcp_server`; the
      callback server serves request_input/report_back; the orchestrator scufris
      server no longer exposes journal/macros/request_input; registration yields
      TWO specs for an orchestrator turn (scufris+den, den gated on den_path)
      and ONE for a sub-agent turn; claude args carry both ids' allowedTools.
- [x] Backend tests for health: `GET /api/agent/mcp` returns scufris+den with
      per-tool enabled flags; den is `warn` when `den_path` unset; a disabled
      tool shows enabled=false and pushes its server to `warn`.
- [x] Frontend test (`web/src/settings-view.test.ts`): grouped render shows one
      details block per server with the right dot, per-tool bulbs reflect
      enabled state, the summary dot aggregates (green/amber/red), and toggling a
      tool still sends the full `disabled_tools` set.
- [x] Docs sync: update `mcp_server.py` module docstring, `AGENTS.md` /
      `README.md` any place that describes "one role-scoped scufris server", and
      add a `CHANGELOG.md` entry. Write `NOTES.md` (as-built + difficulties).

## Definition of Done

- Three servers exist and each serves only its tools: den from
  `den_mcp_server`, callbacks from `agent_mcp_server`, and `mcp_server` holds
  neither (test: `test_servers_expose_disjoint_tool_sets`).
- An orchestrator turn registers `scufris` + `den` (den only when den_path is
  set) and a sub-agent turn registers only the callback server, for BOTH
  backends (test: `test_orchestrator_registers_scufris_and_den` and
  `test_subagent_registers_only_callback_server`).
- `GET /api/agent/tools` returns tools tagged with their real server id
  (`scufris` vs `den`), not all `"scufris"`
  (cmd: `grep -n 'server=' scufris/app.py`) (test:
  `test_agent_tools_tagged_by_server`).
- `GET /api/agent/mcp` returns per-server health with the live-probe statuses:
  den is `warn`/`error` when its den_path is unset or its CLIs are missing, and
  a disabled tool reports `enabled=false` (test:
  `test_mcp_health_den_warn_when_unconfigured`, `test_mcp_health_marks_disabled`).
- The settings page shows an "MCP tools" section with a summary dot that
  expands into per-server collapsibles, each with a status dot and per-tool
  bulbs; toggling a tool still round-trips `disabled_tools` (test:
  `settings-view.test.ts` grouped-render + toggle cases).
- The `apply_role` machinery is gone and nothing references it
  (cmd: `grep -rn --exclude-dir=tasks 'apply_role\|role_tool_names\|_AGENT_ROLE_TOOLS' scufris/ web/`).
- Full gate green: `nix flake check` (ruff + mypy + pytest) and
  `cd web && npm run build && npm test`.
- manual: on the running dashboard, the orchestrator settings show `scufris`
  (green) and `den` (green when the-den is configured, amber otherwise) with
  per-tool bulbs, and a sub-agent's settings show only the callback server.

## Notes

- Relevant files: `scufris/mcp_server.py` (tools + current apply_role at
  899-951, main at 979), `scufris/agent.py` (`scufris_mcp_server` 190-243,
  `_mcp_overrides` 246-298, `_server_override` ~140), `scufris/backends.py`
  (`_scufris_claude_args` 423-468), `scufris/app.py` (tools endpoints
  1767-1859, `_ensure_den_path` 2171, `_ensure_api_base` 2156), `scufris/health.py`
  (in-process MCP check 205-233, `_mcp_tool_count` 81), `scufris/config.py`
  (`disabled_tools` 201, `den_path` 213, `agent_tools_enabled` 196),
  `web/src/settings-view.ts` (`toolCard` 63-86, `renderToolControls` 254-277,
  `healthRow`/`renderHealthCard` 397-440), `web/src/agent-settings-view.ts`
  (data fetch 520-583, render 385-458), `web/src/common.ts` (`AgentTool`,
  `McpServerInfo`), `web/src/style.css` (`health__dot--*` 1851-1911, `tool-grid`
  /`tool-card*` 1707-1756).
- Tool census (31 total): orchestrator agentic 17, den 12 (9 journal_* + 3
  macros_*), sub-agent callbacks 2 (request_input, report_back). The
  orchestrator role already excluded the 2 callbacks, so its visible set is
  unchanged by the move.
- Depends on: 20260727-005013 (CLOSED) - `_ensure_den_path` already bridges
  `SCUFRIS_DEN_PATH` into the dashboard process, which is what lets the den
  live-probe list/exercise den tools in-process.
- Sub-agent callback server id is `agent` (DECISION, confirmed at gate):
  distinct from `scufris` for UI clarity. Steering uses bare tool names so it is
  unaffected; the codex tool prefix becomes `agent.*` and claude sub-agent
  `--allowedTools` becomes `mcp__agent__*`. Update any test asserting
  `mcp__scufris__request_input` / `scufris.request_input`.
- Gate decisions: ONE task (flow single-goal default); land to master (normal
  sprout -> review -> squash-merge, no push).
