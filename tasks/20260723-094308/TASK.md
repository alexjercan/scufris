# BC3: pending_agents() + acknowledge() orchestrator-only MCP tools

- STATUS: OPEN
- PRIORITY: 37
- TAGS: spike,agents,backend,mcp

## Story

As the orchestrator, I want to poll "which agents need me" and clear a signal once
I've handled it, so I can find blocked sub-agents at the end of my own turns even
when the wake bridge is off.

## Context (grounded)

Builds on the durable outcome record (BC1) and the `WAITING` outcomes set by
`request_input` (BC2). Orchestrator-only tools, HTTP-backed like the T2 control
tools (`agent.py:_mcp_overrides` registers the scufris server only for the
orchestrator; `app.py:1106`). Today `agent_status` (`mcp_server.py:237-248`)
exposes only `last_message` per agent, with no "who is waiting" query and no way
to acknowledge/clear.

Spike: `tasks/20260723-001256/SPIKE.md` (BC3).

## Steps (/plan expands)

- [ ] `pending_agents()` MCP tool (orchestrator-only): returns agents with an
      unacknowledged `WAITING`/error outcome and their final message + question
      (HTTP-backed, T2 pattern -> new app read endpoint).
- [ ] `acknowledge(agent_id)` MCP tool (orchestrator-only): clears/acks the
      outcome so it no longer shows up in `pending_agents()`.
- [ ] App endpoints backing both (e.g. `GET /api/agents/pending`,
      `POST /api/agents/{id}/acknowledge`).
- [ ] Keep these OUT of the sub-agent tool set (only `request_input` is
      sub-agent-facing, per BC2).

## Definition of Done

- A `WAITING` (or errored) sub-agent shows up in `pending_agents()` with its
  question; after `acknowledge(agent_id)` it no longer appears.
  (test: `test_pending_agents_then_acknowledge_clears`)
- `pending_agents`/`acknowledge` are registered for the orchestrator only, not
  for sub-agents. (test: tool-set scoping assertion)
- `ruff check .`, `mypy` touched files, `python -m pytest` green from the
  worktree. (cmd: `python -m pytest`)

## Notes

- Depends on BC1 (outcome substrate). Composes with BC2 (the signal source).
- Lessons: `codex-exec-mcp-approval`, the T2 HTTP-backed-tool convention.
- Spike-seeded (BC3).
