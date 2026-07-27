# NOTES - Split MCP into scufris + den + agent servers; per-server live health

As-built record for task 20260727-105609. Decision + rationale live in
`DECISION.md`; this is the design/fix log.

## What changed

### Backend - three single-audience MCP servers

- `scufris/mcp_common.py` (new): the dependency-light shared helpers - `_run`
  (curated-command shell wrapper), `_api_call` + `_api_base` (dashboard HTTP
  bridge), `_disabled_tools`, and `apply_disabled_tools(mcp, names)` (now takes an
  explicit FastMCP so multiple servers can reuse it). No psutil / agent-store /
  FastMCP-at-import, so `den_mcp_server` stays reusable standalone.
- `scufris/den_mcp_server.py` (new): `FastMCP("den")` with the 9 `journal_*` + 3
  `macros_*` tools and `_den_path`/`_journal`. Depends only on `mcp_common` + the
  `today`/`macros` CLIs.
- `scufris/agent_mcp_server.py` (new): `FastMCP("agent")` with `request_input` +
  `report_back` and `_self_agent_id`. Distinct id `agent` (confirmed at the plan
  gate) - steering uses bare tool names, so nothing broke.
- `scufris/mcp_server.py`: trimmed to the 17 orchestrator agentic tools; imports
  the shared helpers; `apply_role`/`role_tool_names`/`ROLE_*`/`_AGENT_ROLE_TOOLS`/
  `_role` all removed (the physical split replaces them). `main()` keeps only the
  disabled-tools filter.
- `scufris/agent.py`: `scufris_mcp_server()` (single) -> `scufris_mcp_servers()`
  (list of `ScufrisMcpServer`, each with a `server_id`). Orchestrator turn ->
  `scufris` + `den` (den only when `den_path` set); sub-agent turn -> `agent`.
  `_mcp_overrides` iterates; `BUILTIN_MCP_SERVER_IDS = {scufris, den, agent}`
  guards operator `mcp_servers` collisions.
- `scufris/backends.py`: `_scufris_claude_args` builds one `mcpServers` entry and
  one `--allowedTools mcp__<id>__*` wildcard per registered server.
- `scufris/app.py`: `_as_agent_tool` now takes the real server id; `/api/agent/tools`
  and `/api/agents/{id}/tools` aggregate the audience's servers via
  `_mcp_servers_for_audience` (delegates to `mcp_health.servers_for_audience`);
  `run_agent_tool` resolves the tool across the orchestrator's servers. New
  `McpServerHealth` model + `GET /api/agent/mcp` and `/api/agents/{id}/mcp`.
- `scufris/mcp_health.py` (new): the in-process live probe - `servers_for_audience`
  + `probe_server` (list_tools + den readiness: den_path set + `today`/`macros`
  on PATH). Returns `(status, detail, available, tools)`.
- `scufris/health.py`: `_mcp_tool_count` now sums scufris + den.

### Frontend - grouped "MCP tools" section

- `common.ts`: `AgentTool.available?` + `McpServerHealth`.
- `settings-view.ts`: `renderToolControls` replaced by `renderMcpServers(servers,
  actions|null)` - a summary dot over one `<details>` block per server, each a
  `tool-grid` of cards with a green/red bulb; `actions` null = read-only (a
  sub-agent), set = writable (toggle + runner, orchestrator). Reuses `toolCard`,
  `toolControlCard`, `toolRunner`, and the `health__dot--{ok,warn,error}` styles.
- `agent-settings-view.ts`: fetches `/api/agent/mcp` (orchestrator) or
  `/api/agents/{id}/mcp` (sub-agent) into `data.mcpServers`; renders the grouped
  view in place of the old flat orchestrator Tools grid and the sub-agent
  `agentToolsPanel` (both removed). `AgentSettingsGlobal.tools` and
  `AgentSettingsData.agentTools` dropped.
- `style.css`: `.mcp-server*`, `.settings__title-row`, `.tool-card__bulb`.

## Difficulties / decisions along the way

- "Live probe" was confirmed as spawn-each-server in the plan gate, but the whole
  dashboard already lists/runs MCP tools IN-PROCESS (and `health.py` already probes
  via `list_tools()`). Spawning stdio subprocesses just for the settings health
  would be inconsistent and costly, and the in-process import catches the same real
  failures (import error, missing `today`/`macros`, den unset). Realised it
  in-process and recorded that in DECISION.md rather than silently diverging.
- The `scufris_mcp_server` single -> list rename hit every caller and test double
  (agent.py, backends.py, app.py x3, plus test_agent/test_backends). Grepped all
  callers up front (`protocol-signature-change-hits-the-doubles`) so the sweep was
  one pass, not a trail of `TypeError`s.
- Per-tool bulb semantics: a disabled tool makes its server amber but leaves the
  OTHER tools available; a den that can't run (unset/CLI-missing) makes ALL its
  tools unavailable. So the probe returns a per-server `available` flag distinct
  from per-tool `enabled`.
- Test hermeticity: the claude backend tests used bare `Settings()`, which on a
  dev box reads `.env` and could add a `den` server; switched them to
  `_env_file=None` (and the health tests `monkeypatch.delenv("SCUFRIS_DEN_PATH")`)
  so a configured dev den can't redden them (`settings-test-must-disable-env-file`).

## Self-reflection

- The test split (one file per new module) was the right call - the old
  `test_mcp_server.py` was doing three modules' jobs. Did it by moving tests
  verbatim and only re-pointing the `_run` monkeypatch target, so behaviour
  coverage is unchanged.
- Biggest risk was the large `mcp_server.py` deletion; used a range `sed` for the
  bulk cut then re-read + ran ruff/mypy/import-smoke immediately, so a mis-cut would
  have surfaced at once rather than deep in the test run.
