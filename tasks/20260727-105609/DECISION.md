# Decision: Split the one role-scoped scufris MCP into three single-audience servers

- DATE: 20260727-105609
- STATUS: ACCEPTED
- TASK: 20260727-105609
- TAGS: decision, mcp, agents

## Context

Today a single `FastMCP("scufris")` in `scufris/mcp_server.py` holds all 31
tools and serves two audiences via a runtime filter (`apply_role` reads
`SCUFRIS_AGENT_ROLE`): the orchestrator gets everything except the sub-agent
callbacks; a sub-agent gets ONLY `request_input` + `report_back`. The tools
span three unrelated concerns: orchestrator "agentic" control (host/observe/
projects/agents), the operator's "life" domain (journal + macros, gated on
`SCUFRIS_DEN_PATH`), and the sub-agent comms callbacks. The operator asked to
(a) physically split these and (b) surface each server with a live health
light and per-tool bulb in settings. Two forks were confirmed with the user:
the split MECHANISM and what the health light MEANS.

## Decision

Split into THREE separate modules, each its own `FastMCP` instance and each a
SINGLE audience:

1. `scufris/mcp_server.py` - id `scufris`, orchestrator agentic (17 tools:
   host_stats, disk_usage, list_processes, list_agents, agent_status, the 5
   project CRUD, the 5 agent control, pending_agents, acknowledge).
2. `scufris/den_mcp_server.py` - id `den`, life (12 tools: 9 `journal_*` + 3
   `macros_*`). Depends only on the `today`/`macros` CLIs and a lean shared
   shell helper - never on the dashboard API, psutil, or the agent store, so it
   is reusable standalone.
3. `scufris/agent_mcp_server.py` - id `agent` (a DISTINCT id so the "3rd MCP"
   reads clearly in the UI and the audience is unmistakable), sub-agent
   callbacks (2 tools: request_input, report_back). Steering references the
   callbacks by BARE name so it is unaffected; the model-facing ids become
   `agent.request_input` (codex) / `mcp__agent__request_input` (claude) and the
   claude sub-agent `--allowedTools` wildcard becomes `mcp__agent__*`.

A turn registers only the servers for its audience: an ORCHESTRATOR turn gets
`scufris` + `den` (den only when `den_path` is set); a SUB-AGENT turn gets only
the callback server. The audience boundary is therefore PHYSICAL (a sub-agent
turn never registers the orchestrator/den servers) rather than a filter inside
one server, so `apply_role`/`role_tool_names`/`_AGENT_ROLE_TOOLS` are retired.
`apply_disabled_tools` stays on the two orchestrator servers.

Health is a LIVE PROBE realised IN-PROCESS: for each server the dashboard
imports that module's `mcp`, calls `list_tools()`, and runs the server's real
readiness checks (den: `den_path` configured + `today`/`macros` on PATH;
scufris: tool count > 0). This mirrors how the dashboard already lists and runs
tools in-process (`/api/agent/tools`, the "try it" runner) and how `health.py`
already probes MCP via `list_tools()`. A per-server dot is green (loaded +
ready + all tools enabled), amber (loaded but degraded - some tools disabled,
or den configured-but-CLI-missing / den_path unset), or red (import/list
failed). A per-tool bulb is green (advertised + enabled) or red (disabled or
its server unavailable).

## Alternatives considered

- **Same binary, new `den` role (rejected).** Keep one `mcp_server.py` and add
  a `den` value to `SCUFRIS_AGENT_ROLE` so `apply_role` serves only the life
  tools. Lightest change, but the "separation" stays a runtime filter, den is
  not independently reusable (still imports the whole module with psutil / the
  agent store / the dashboard API), and the audience boundary stays a filter
  rather than a hard "not registered" guarantee.
- **Health derived from config, no probe (rejected).** Compute status purely
  from the known tool set minus `disabled_tools` plus a `den_path` check. Fast
  and simple, but it cannot catch a genuinely broken server (import error, a
  missing `today`/`macros` CLI) - which is exactly the "red if they fail" the
  operator asked for.
- **Health via stdio-subprocess spawn (rejected).** Spawn each server the way a
  turn does and drive the MCP handshake. Most faithful to a real turn, but it
  is inconsistent with the whole dashboard already treating the tools
  in-process, adds per-load subprocess cost, and the in-process import catches
  the same real failure modes (import error, missing CLI, den unset).
- **Keep the sub-agent server id as `scufris` (rejected).** Lowest churn (tool
  ids and the claude allowedTools wildcard unchanged), but reusing one id for
  two different modules/audiences muddies the "3 MCPs" model the operator wants
  to see; a distinct `agent` id was chosen for clarity (the churn is a
  bare-name-safe rename of one wildcard and the codex tool prefix).

## Consequences

- Isolation gets STRONGER and simpler to reason about: a sub-agent turn cannot
  reach orchestrator/den tools because those servers are never registered on
  it, not because a filter removed them. The `apply_role` machinery and its
  tests go away.
- Three module entrypoints (`python -m scufris.{mcp_server,den_mcp_server,
  agent_mcp_server}`) and a small shared `scufris/mcp_common.py` (the `_run`
  shell wrapper, `_api_call`, output/timeout caps) instead of one file. The
  registration core (`agent.scufris_mcp_server`) now returns a LIST of server
  specs, and both backends (codex `-c`, claude `--mcp-config`) iterate it.
- `AgentTool.server` becomes meaningful (was hardcoded `"scufris"`); the tools
  endpoints must tag each tool with its real server and aggregate across the
  orchestrator's two servers.
- The settings "MCP tools" section is driven by a new per-server health
  endpoint; a sub-agent's settings page shows its one callback server, the
  orchestrator's shows `scufris` + `den`.
- Cost: den probe shells out to `shutil.which("today"/"macros")` per health
  fetch (cheap); the health fetch imports both server modules in the dashboard
  process (already imported for the tools console).
