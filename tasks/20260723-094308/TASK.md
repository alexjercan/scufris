# BC3: pending_agents() + acknowledge() orchestrator-only MCP tools

- PRIORITY: 37
- TAGS: spike, agents, backend, mcp
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] `pending_agents()` MCP tool (orchestrator-only): GETs
      `/api/agents/pending` and renders a row per waiter (id, state, message).
      Backed by `AgentStore.pending_outcomes()` (unacknowledged WAITING/ERROR).
- [x] `acknowledge(agent_id)` MCP tool (orchestrator-only): POSTs
      `/api/agents/{id}/acknowledge`; `AgentStore.acknowledge` sets the outcome's
      `acknowledged` flag (idempotent, never raises for an unknown/cleared agent).
- [x] App endpoints: `GET /api/agents/pending` (declared BEFORE
      `/api/agents/{id}` so "pending" is not parsed as an id, mirroring
      `/api/agents/backends`) and `POST /api/agents/{id}/acknowledge`.
- [x] These are orchestrator-audience automatically under the BC2 role model
      (not in `_AGENT_ROLE_TOOLS`), so `apply_role(agent)` removes them - the
      agent role still exposes ONLY `request_input`.

## Definition of Done

- A `WAITING` (or errored) sub-agent shows up in `pending_agents()`/
  `/api/agents/pending` with its question; after `acknowledge(agent_id)` it no
  longer appears.
  (test: `test_acknowledge_clears_from_pending` (store);
  `test_pending_agents_and_acknowledge_roundtrip` (app endpoints);
  `test_pending_agents_formats_the_poll` / `test_acknowledge_posts_to_the_endpoint`
  (tools))
- `pending_agents`/`acknowledge` are registered for the orchestrator only, not
  for sub-agents.
  (test: `test_apply_role_agent_keeps_only_request_input` - the agent role's tool
  set is exactly `{request_input}`, so the BC3 tools are absent there;
  `test_tools_registered` includes them in the full set)
- `ruff check .`, `mypy`, `python -m pytest` green from the worktree.
  (cmd: `python -m pytest`)

## Notes

- Depends on BC1 (outcome substrate). Composes with BC2 (the signal source).
- Lessons: `codex-exec-mcp-approval`, the T2 HTTP-backed-tool convention,
  `test-the-net-new-route-not-the-reused-path` (each new endpoint tested directly).
- Spike-seeded (BC3).

## Close record (2026-07-23)

What changed:
- `agent_store.py`: `pending_outcomes()` (unacknowledged WAITING/ERROR outcomes,
  a cleanly DONE agent is not pending) and `acknowledge(agent_id) -> bool` (flips
  the outcome's `acknowledged` flag via `model_copy`, idempotent, no raise for an
  unknown/cleared agent).
- `mcp_server.py`: `pending_agents()` (GET `/api/agents/pending`, renders a
  table) and `acknowledge(agent_id)` (POST). Both orchestrator-audience - not in
  `_AGENT_ROLE_TOOLS`, so `apply_role(agent)` strips them automatically (the BC2
  role model paid its dividend: zero new scoping plumbing).
- `app.py`: `GET /api/agents/pending` (list of `PendingAgent`, newest first) and
  `POST /api/agents/{id}/acknowledge` (`AcknowledgeResult`). The pending route is
  declared BEFORE `/api/agents/{id}` so "pending" is not parsed as an agent id -
  the same guard `/api/agents/backends` already uses. CHANGELOG Added entry.

Evidence: red-first tests at each layer - store (pending lists WAITING+ERROR not
DONE/acknowledged; acknowledge clears + persists + idempotent), tools (respx:
pending_agents formats/empty, acknowledge posts/validates), app (the
request_input -> pending -> acknowledge -> empty round-trip through the real
routes, proving the static /pending route is not shadowed). Suite 367 passed
(360 baseline + 7); ruff + mypy clean.

Design note: the "pending" set is unacknowledged WAITING or ERROR; DONE is not
pending (a clean finish needs nothing). `acknowledge` is idempotent and 404-free
- a deleted agent's outcome is already cleared by `delete`, so acking it is a
harmless False. The static-vs-param route-ordering trap (a `GET /api/agents/pending`
shadowed by `/api/agents/{id}`) was avoided by following the existing
`/api/agents/backends` precedent.

Self-reflection: this task was the easy dividend of BC2's DECISION - because the
role model scopes by audience, the two orchestrator tools needed no scoping work
at all; the only real design choice was the pending-set predicate and the route
ordering, both small. Building the store predicate first meant the tool/endpoint
layers were thin.
