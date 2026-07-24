# Goal: parent-session routing for sub-agent escalations (part 3, redefined)

- DATE: 20260724
- UMBRELLA TASK: 20260724-132713
- LANDING SCOPE: squash-merge to master (local, no push), as in parts 1-2.

## Goal

Redefines the seeded part-3 task (20260724-111959) after review: recording a bare
`parent_agent_id` is a no-op (only the orchestrator spawns, and `request_input`
already reaches it via the global `pending_agents` poll). The valuable version,
enabled by part 1's multi-session orchestrator, is to record which orchestrator
CHAT (session) spawned each child and route escalations back to that chat, so a
child that calls `request_input` surfaces to the specific orchestrator
conversation that launched it - not to every chat.

Mechanism (see DECISION.md): inject the orchestrator's current session id into
its per-turn MCP server env as `SCUFRIS_ORCH_SESSION_ID`; `message_agent` /
`run_agent` read it and pass `parent_session_id` (+ parent_agent_id =
orchestrator) when they spawn a child; the run/chat endpoints persist those on
the child's registry entry; `pending_agents` reads the same env and asks the
pending endpoint to attribute/filter waiting children by the calling chat.

Known edge (documented, accepted): a child spawned during the orchestrator's very
first turn of a brand-new chat has no session id yet, so its `parent_session_id`
is empty and it shows as unattributed (visible to all chats) rather than being
orphaned.

## Done means

1. The orchestrator's MCP env carries `SCUFRIS_ORCH_SESSION_ID` = its current
   session id on a resumed turn (test: `test_orchestrator_mcp_env_has_session_id`).
2. `message_agent` / `run_agent` pass `parent_session_id` from that env when
   spawning a child (test: `test_message_agent_forwards_parent_session`).
3. The child's registry entry records `parent_agent_id` + `parent_session_id` at
   spawn (test: `test_spawn_records_parent_on_child`).
4. `pending_agents` attributes each waiting child to its parent chat and a chat
   sees its own children (+ unattributed), not another chat's
   (test: `test_pending_filtered_by_parent_session`).
5. request_input end-to-end still works (existing comms tests still pass; e.g.
   `examples/comms_loop.py` path / existing pending tests).

Overall: `nix flake check` green (ruff + mypy + pytest).

## Tasks

- [ ] 20260724-132830 (p38, scufris) Parent-session routing: attribute + route sub-agent escalations to the spawning chat

## Decisions (load-bearing, architectural)

- 20260724-132713 DECISION.md: attribute escalations to the spawning chat via
  SCUFRIS_ORCH_SESSION_ID; filter pending with an unattributed fallback (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (pending) end-to-end: with two orchestrator chats each spawning a sub-agent,
  a child's request_input surfaces to the chat that spawned it.
