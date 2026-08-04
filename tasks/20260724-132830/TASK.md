# Parent-session routing: attribute + route sub-agent escalations to the spawning orchestrator chat

- PRIORITY: 38
- TAGS: agents, sessions, comms
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

With multiple orchestrator chats (part 1), a sub-agent that calls
`request_input` currently surfaces to EVERY chat's `pending_agents` poll. Record
which orchestrator chat spawned each child and route the escalation back to that
chat. Supersedes the seeded no-op task 20260724-111959 (a bare `parent_agent_id`
adds no signal). Mechanism + edges: tasks/20260724-132713/DECISION.md. Umbrella:
20260724-132713.

## Steps

- [x] `SessionRegistry` (`agent_store.py`): add `parent_session_id` to the entry
      (alongside the reserved `parent_agent_id`); `_load` tolerant (default None);
      add `set_parent(agent_id, parent_agent_id, parent_session_id)` (creates a
      minimal entry when the child has no session yet) and `parent_of(agent_id)
      -> tuple[str|None, str|None]`.
- [x] `AgentStore`: `record_spawn_parent(child_id, parent_agent_id,
      parent_session_id)` delegating to the registry; expose `parent_of`.
- [x] Env: thread the orchestrator's resumed session id into `scufris_mcp_server`
      (`agent.py`) as `SCUFRIS_ORCH_SESSION_ID` in the orchestrator env branch.
      Source it from the turn's `session_id`/`thread_id`: codex via
      `_mcp_overrides`/`_stream_app_server`, claude via `_scufris_claude_args`/
      `ClaudeBackend.stream`. Empty on a fresh turn (no resumed id). No new value
      to compute - it is the id the turn is already resuming.
- [x] MCP tools (`mcp_server.py`): `message_agent` and `run_agent` read
      `SCUFRIS_ORCH_SESSION_ID` (a `_orch_session_id()` helper, mirroring
      `_self_agent_id`) and include `parent_session_id` in their POST body to
      `/chat` / `/run`.
- [x] Endpoints (`app.py`): `AgentChatRequest` / `AgentRunRequest` gain optional
      `parent_session_id`; `agent_chat` / `run_agent` call
      `agents.record_spawn_parent(agent_id, ORCHESTRATOR_ID, parent_session_id)`
      before launching when a `parent_session_id` is supplied.
- [x] Routing (`app.py` `list_pending_agents` + `PendingAgent`): accept an
      optional `parent_session_id` query; enrich each row with the child's
      `parent_agent_id`/`parent_session_id` (from `parent_of`); when the query is
      given, return children whose parent session == it OR is empty
      (unattributed), never another chat's. `pending_agents()` MCP tool passes the
      env session id and renders the parent chat in its table.
- [x] Tests: env carries the session id on a resumed orchestrator turn;
      message_agent/run_agent forward `parent_session_id`; spawn records parent on
      the child; pending filters by parent session with the unattributed
      fallback; existing comms/pending tests still pass.
- [x] NOTES.md; the DECISION.md is already written - index it in GOAL.md.

## Definition of Done

- Resumed orchestrator turn's MCP env has `SCUFRIS_ORCH_SESSION_ID` = its session
  (test: `test_orchestrator_mcp_env_has_session_id`); fresh turn -> empty
  (same test's second case).
- `message_agent`/`run_agent` forward `parent_session_id` from the env
  (test: `test_message_agent_forwards_parent_session`).
- A spawn records `parent_agent_id`+`parent_session_id` on the child
  (test: `test_spawn_records_parent_on_child`).
- `pending_agents` filtered by a chat returns that chat's children + unattributed,
  not another chat's (test: `test_pending_filtered_by_parent_session`).
- Existing comms/pending behavior intact (existing tests + `examples/comms_loop.py`
  path unchanged; cmd: `python -m pytest tests -k "pending or comms or request_input"`).
- Full QA gate green (cmd: `nix flake check`).

## Notes

- Decision: tasks/20260724-132713/DECISION.md. Spike: tasks/20260724-111839/SPIKE.md.
- Relevant files: `scufris/agent_store.py` (SessionRegistry + AgentStore),
  `scufris/agent.py` (`scufris_mcp_server`, `_mcp_overrides`,
  `_stream_app_server`), `scufris/backends.py` (`_scufris_claude_args`,
  `ClaudeBackend.stream`), `scufris/mcp_server.py` (`message_agent`, `run_agent`,
  `pending_agents`, add `_orch_session_id`), `scufris/app.py`
  (`AgentChatRequest`/`AgentRunRequest`, `agent_chat`, `run_agent`,
  `list_pending_agents`, `PendingAgent`).
- Backward-compat: no `parent_session_id` (UI-spawned, or older env) -> child is
  unattributed -> visible to all chats, exactly today's behavior. The single-chat
  flow is unchanged.
- SUPERSEDES 20260724-111959 (close it when this lands).

## Outcome (CLOSED)

A sub-agent's `request_input` now routes to the orchestrator chat that spawned it.
`SCUFRIS_ORCH_SESSION_ID` (the resumed session id) rides the orchestrator MCP env
-> `message_agent`/`run_agent` send `parent_session_id` -> the child's registry
entry records (orchestrator, chat) -> `pending_agents` scopes to the calling chat
(own children + unattributed, never another chat's). See DECISION.md (umbrella)
and NOTES.md.

- All Steps landed; every DoD proof passes; 448 pytest + ruff + mypy +
  `nix flake check` green; 19 existing comms/pending tests intact.
- A/B on the env injection and the pending filter both go red when neutered.
- Fresh-turn edge (unattributed until the first turn finishes) documented, not a
  bug. Back-compat: no parent -> visible to all chats (the old single-chat flow).
