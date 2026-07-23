# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The per-agent page (`/agents/<id>`) now reattaches to an in-flight turn on
  load. Its chat used to only rebuild the settled transcript and stream turns the
  browser itself POSTed, so a turn driven from elsewhere (the orchestrator's
  `message_agent`/`run_agent` against a sub-agent, which runs on the shared
  supervisor + event bus) never showed live, and reloading/reselecting mid-turn
  froze on the settled transcript. On mount it now subscribes to
  `GET /api/agents/<id>/events` (gated on an active run so a finished run is not
  replayed as a phantom bubble), streams the in-flight turn to completion, and
  settles the streamed reply into the log (the turn's prompt line comes from the
  mount-time transcript). Restores the SSE reattach the detail-page reshape had
  dropped.
- Agent session ids now live in a persisted, backend-tagged registry
  (`<state_dir>/sessions.json`) keyed by agent id - for ALL agents, the landing
  orchestrator included. The orchestrator's session used to be in-memory only, so
  a server restart lost its conversation and left read paths free to latch onto a
  sub-agent's codex rollout (the observed orchestrator/sub-agent transcript
  mixing). Deleting an agent removes its mapping; switching an agent's backend
  clears the stale wrong-backend id; a legacy `agents.json` `session_id` migrates
  into the registry on first load.

### Removed

- The `tatr_ls`, `tatr_show` and `tatr_new` MCP tools. The orchestrator manages
  tatr tasks with the `tatr` skill via `Bash`, so a dedicated MCP wrapper is
  redundant. The host/observe tools
  (`host_stats`, `disk_usage`, `list_processes`, `list_agents`, `agent_status`) and
  the new control tools remain; the tool-steering preamble no longer mentions tatr.

### Changed

- The landing orchestrator's permission mode now DEFAULTS to `auto` (edit + run
  commands) instead of `manual` (read-only) - it does write work unattended (Bash
  tatr, create projects/agents). Editable at runtime from its settings page or via
  `SCUFRIS_AGENT_PERMISSION_MODE`; project agents are unaffected (their records
  still default to manual).

- The built-in `scufris` MCP server is now ROLE-SCOPED: the landing orchestrator's
  turns get the full surface (host/observe/control tools and the tool-steering
  preamble), while regular project agents get ONLY the `request_input` callback
  (see Added) - not the full toolset they used to receive. They draw the rest of
  their tools from their own project config/skills. This threads an
  `is_orchestrator` role and the agent's own id through the backend `stream` path;
  operator-declared `mcp_servers` still apply to every agent.

### Added

- Claude backend reaches scufris MCP parity with codex: a claude-backed agent now
  gets the built-in role-scoped `scufris` server wired into every turn via
  `--mcp-config` (an inline JSON blob) + `--strict-mcp-config` + `--allowedTools
  mcp__scufris__*`, so a claude sub-agent can call `request_input` (and the
  orchestrator its control tools) unattended - the full comms loop self-heals on
  claude, not just codex. The role env (`SCUFRIS_AGENT_ROLE` / `SCUFRIS_AGENT_ID` /
  `SCUFRIS_DISABLED_TOOLS`) now comes from a backend-agnostic `scufris_mcp_server`
  core that both backends format to their own flavour (codex to `-c` overrides,
  claude to the JSON config), so they cannot drift on what a role exposes. The
  whole-server `mcp__scufris__*` allowlist is role-safe because the server enforces
  the role scope itself.
- Role-scoped per-agent tools view: `GET /api/agents/{id}/tools` returns the tools
  an agent can actually call in its turns - the orchestrator's full surface, a codex
  or claude sub-agent's `request_input` only, and NOTHING for a backend that does not
  wire the scufris MCP (opencode/mock, today) - instead of the global unscoped set the
  UI used to show. Each project agent's settings page now renders a read-only Tools
  card from it, so a sub-agent shows its real tool surface (one tool, not the
  orchestrator's eighteen). The orchestrator keeps its writable operator console
  (`/api/agent/tools`), which stays the full in-process set.
- A runnable end-to-end example ([`examples/comms_loop.py`](examples/comms_loop.py))
  and an acceptance test (`test_stalled_merge_loop_self_heals`, parametrized on
  both wake paths) that replay the stalled-merge scenario against the mock backend:
  a sub-agent blocks (`request_input`), the orchestrator is woken (bridge) or polls
  (`pending_agents`), answers by resuming the sub-agent's session, and the loop
  resolves - proving the bidirectional-comms feature self-heals the case the spike
  exists to fix, not just its pieces (spike 20260723-001256).
- Auto-wake bridge (opt-in via `SCUFRIS_AUTO_WAKE`, off by default): when a
  sub-agent finishes a run awaiting a decision (a `WAITING` outcome from
  `request_input`) or errors, the dashboard grants the orchestrator a turn with the
  question injected, so a stalled loop self-heals without the operator driving it.
  Wakes are deferred while the orchestrator is mid-turn and batched into one turn
  when it goes idle - never dropped, and the waker never holds the orchestrator's
  serialize key. When off, the orchestrator polls `pending_agents` (BC3) instead.
  Completes bidirectional agent<->orchestrator comms (spike 20260723-001256).
- Sub-agents can signal the orchestrator that they are blocked and need a
  decision, via a `request_input` MCP tool - the only scufris tool a regular agent
  gets (see the role scoping under Changed). Calling it records a WAITING outcome
  carrying the question, preserved across the agent's turn-end (so the natural
  completion does not clobber it) - the orchestrator answers later by resuming the
  session. Wired on both the codex and claude backends (see the claude MCP-parity
  entry above). Part of bidirectional agent<->orchestrator comms
  (spike 20260723-001256).
- Orchestrator-only `pending_agents` and `acknowledge` MCP tools (and the
  `GET /api/agents/pending` / `POST /api/agents/{id}/acknowledge` endpoints behind
  them): the orchestrator can poll "which sub-agents need me" - those with an
  unacknowledged `request_input` (WAITING) or ERROR outcome, with their question -
  and clear one once handled, so a blocked sub-agent no longer waits forever.
  Part of bidirectional agent<->orchestrator comms (spike 20260723-001256).
- A durable per-agent run-outcome record (`<state_dir>/outcomes.json`): when a
  run ends, the final message and terminal state are persisted for every agent,
  so the orchestrator can observe a finished agent AFTER its per-run event stream
  has closed - the substrate for bidirectional agent<->orchestrator comms
  (spike 20260723-001256). A new `AgentState.WAITING` ("ended a turn awaiting a
  decision") names the needs-input state, distinct from `BLOCKED` (waiting on an
  approval). Deleting an agent drops its outcome.
- Full CRUD orchestrator control tools on the scufris MCP server: `get_project`,
  `update_project`, `delete_project`, `update_agent` and `delete_agent` join the
  existing create/list/run/message tools, so the orchestrator can edit an agent's
  permission mode (manual|edit|auto), provider (codex|claude) and model, and manage
  projects, all from chat. The PATCH tools send only the fields you pass. The agent
  write tools edit REGULAR agents only - the reserved orchestrator configures itself
  via settings and is refused.
- Orchestrator control tools on the scufris MCP server (orchestrator-only): the
  landing orchestrator can now DO dashboard actions, not just observe. `list_projects`,
  `create_project`, `create_agent`, `run_agent` and `message_agent` call the
  dashboard's own HTTP API at `SCUFRIS_API_BASE` (127.0.0.1:<port>, injected when the
  dashboard spawns the server), reusing each endpoint's validation and lifecycle since
  the MCP subprocess cannot touch the live in-app supervisor. Curated and bounded like
  the existing tools; a non-2xx or network failure returns `error:` text, never an
  exception.
- Settings page: an interactive "try it" runner on each enabled tool card - reveal
  a form generated from the tool's parameter schema, confirm, and run one MCP tool
  in isolation with its result rendered inline, without a chat turn. Backed by a new
  `POST /api/agent/tools/{name}/run` endpoint that runs a single scufris tool
  in-process (bypassing the agent) and refuses a disabled tool (403), an unknown tool
  (404), or bad args (422). The tools listing (`GET /api/agent/tools`) now also
  exposes each tool's typed parameter schema.
