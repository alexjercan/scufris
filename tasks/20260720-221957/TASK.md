# A5: orchestrator observation MCP tools (list_agents, agent_status)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: spike,agents

## Goal

Orchestrator observation (read-only, v1): give the main chat agent MCP tools
`list_agents` and `agent_status(id)` (built on the A2 status contract) so I can
ask the orchestrator "what is agent-N working on" and it answers by reading that
agent's status. No steering in v1 - observe + report only.

## Steps

- [ ] Add two read-only tools to `scufris/mcp_server.py`: `list_agents()` and
      `agent_status(agent_id)`. They run in the MCP SUBPROCESS (spawned by
      codex), so they read PERSISTED state - `AgentStore` from `agents.json`
      (A3's `mark_running`/`mark_finished` persist the lifecycle) + the backend's
      `read_status` from the rollout/session files. No supervisor access (that
      lives in the app process), and NO steering (read-only, decision 4/v1).
- [ ] Factor the logic into pure helpers `_list_agents_text(settings)` /
      `_agent_status_text(settings, agent_id)` (the `@mcp.tool()` wrappers just
      call them with `Settings()`), so tests drive them with a temp state dir
      without env monkeypatching. `list_agents` -> a compact table (id, state,
      backend, project, name); `agent_status` -> the agent's config + state +
      the backend `read_status` progress (turns, tokens, last message), a clear
      "no such agent" message on a bad id, and a graceful line if status is
      unreadable.
- [ ] Tests (`tests/test_mcp_server.py`): `_list_agents_text` over a seeded
      `agents.json` (mock backend) formats the rows + an empty-state message;
      `_agent_status_text` returns state + progress for a mock agent (session id
      set) and a "no such agent" for an unknown id; both tools appear in the MCP
      catalog (`mcp.list_tools()`).
- [ ] Full check suite green; close-out.

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
