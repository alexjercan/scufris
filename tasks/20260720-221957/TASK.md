# A5: orchestrator observation MCP tools (list_agents, agent_status)

- PRIORITY: 20
- TAGS: spike, agents
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Orchestrator observation (read-only, v1): give the main chat agent MCP tools
`list_agents` and `agent_status(id)` (built on the A2 status contract) so I can
ask the orchestrator "what is agent-N working on" and it answers by reading that
agent's status. No steering in v1 - observe + report only.

## Steps

- [x] Add two read-only tools to `scufris/mcp_server.py`: `list_agents()` and
      `agent_status(agent_id)`. They run in the MCP SUBPROCESS (spawned by
      codex), so they read PERSISTED state - `AgentStore` from `agents.json`
      (A3's `mark_running`/`mark_finished` persist the lifecycle) + the backend's
      `read_status` from the rollout/session files. No supervisor access (that
      lives in the app process), and NO steering (read-only, decision 4/v1).
- [x] Factor the logic into pure helpers `_list_agents_text(settings)` /
      `_agent_status_text(settings, agent_id)` (the `@mcp.tool()` wrappers just
      call them with `Settings()`), so tests drive them with a temp state dir
      without env monkeypatching. `list_agents` -> a compact table (id, state,
      backend, project, name); `agent_status` -> the agent's config + state +
      the backend `read_status` progress (turns, tokens, last message), a clear
      "no such agent" message on a bad id, and a graceful line if status is
      unreadable.
- [x] Tests (`tests/test_mcp_server.py`): `_list_agents_text` over a seeded
      `agents.json` (mock backend) formats the rows + an empty-state message;
      `_agent_status_text` returns state + progress for a mock agent (session id
      set) and a "no such agent" for an unknown id; both tools appear in the MCP
      catalog (`mcp.list_tools()`).
- [x] Full check suite green; close-out.

## Definition of Done

- `list_agents` lists the configured agents with their state (test:
  `list_agents_text_formats_rows`).
- `agent_status(id)` reports an agent's state + progress, and errors clearly on
  an unknown id (test: `agent_status_text_reports_progress`,
  `agent_status_text_unknown_id`).
- Both tools are registered on the scufris MCP server (test:
  `orchestrator_tools_in_catalog`).
- The full suite passes (cmd: `nix develop --command bash -c "ruff check . &&
  mypy . && pytest -q"`).
- manual: ask the orchestrator "what is agent-N working on" and get a correct,
  read-only answer (batched for Finish; needs a live codex orchestrator turn).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 4; steering deferred).
- Depends on: 20260720-221935 (A2, landed 4d6850a); reads A1's store + A3's
  persisted lifecycle.
- Cross-process by design: the MCP tools read the SAME persisted files the app
  writes (agents.json, rollouts), so no shared memory is needed. Live
  supervisor-only state (queued vs running before the first persist) is not
  visible here - the record's `state` is the observable, which is enough for
  "what is agent-N doing".

## Close-out

What changed:
- `scufris/mcp_server.py`: two read-only orchestrator tools - `list_agents()`
  (compact table: id, state, backend, project, name) and `agent_status(agent_id)`
  (config + lifecycle state + backend read_status progress: turns, tool calls,
  tokens, last message; clear "no such agent"; graceful line if progress
  unreadable). Logic factored into pure helpers `_list_agents_text(settings)` /
  `_agent_status_text(settings, agent_id)`; the tool wrappers call them with
  `Settings()`. A `TYPE_CHECKING` block names the lazily-imported types.
- Tests (`tests/test_mcp_server.py`): both helpers over a seeded state dir (mock
  backend) - list formats rows + empty-state; status reports progress (incl. a
  cross-process re-read after `mark_finished` persisted the session) + unknown-id
  error; both tools in the MCP catalog (`test_tools_registered` updated).

Design:
- Cross-process by design: the tools run in the MCP SUBPROCESS (spawned by
  codex), which cannot see the app's in-memory Supervisor. They read the SAME
  persisted files the app writes - `agents.json` (the run engine persists the
  lifecycle via mark_running/mark_finished) and the rollout/session files (via
  the backend read_status). The record's `state` is the observable; no shared
  memory needed.
- READ-ONLY (decision 4 / v1): observe + report, never launch or steer. Steering
  is a deliberate later phase.
- Imports of AgentStore/backends stay LAZY (inside the helpers) to keep the MCP
  server's startup import light; `TYPE_CHECKING` gives mypy the names.

Result: 250 tests pass (+4), ruff + mypy clean.

Self-reflection: the whole task was small because A1 (store) + A3 (persisted
lifecycle) + A2 (read_status) already provided everything; A5 is just a
read-only view over persisted state. The one real design point - cross-process
via the persisted files, not shared memory - fell straight out of the earlier
"codex owns the rollout, scufris reads it" architecture.
