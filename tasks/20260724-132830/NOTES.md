# Notes: parent-session routing for sub-agent escalations (part 3)

- TASK: 20260724-132830
- BRANCH: feature/parent-session-routing

## What changed

A sub-agent's `request_input` now surfaces to the specific orchestrator chat that
spawned it, not to every chat. Flow:

1. **Capture (no new value to compute).** The orchestrator's current chat is the
   session its turn is resuming. `scufris_mcp_server` (`agent.py`) injects it into
   the orchestrator MCP env as `SCUFRIS_ORCH_SESSION_ID`, threaded from the turn's
   `thread_id`/`session_id` via `_mcp_overrides`/`_stream_app_server` (codex) and
   `_scufris_claude_args`/`_claude_stream_args` (claude). Empty on a fresh turn.
2. **Propagate.** `message_agent`/`run_agent` (`mcp_server.py`) read it via a new
   `_orch_session_id()` and send `parent_session_id` to `/chat`/`/run`; those
   endpoints (`app.py`) call `agents.record_spawn_parent(child, orchestrator, ps)`.
3. **Persist.** `SessionRegistry` (`agent_store.py`) gained `parent_session_id`
   next to the reserved `parent_agent_id`, with `set_parent`/`parent_of`. Parent is
   a backend-independent fact: `set_parent` works before the child has a session
   (a placeholder entry), and `_fresh` preserves it across a backend switch.
4. **Route.** `GET /api/agents/pending?parent_session_id=X` returns children whose
   parent session == X OR is empty (unattributed), never another chat's, and
   annotates each row with its parent. `pending_agents()` passes the env chat.

## Why / design

See DECISION.md (umbrella): a bare `parent_agent_id` was a no-op (only the
orchestrator spawns, and pending was already a global poll); part 1's
multi-session is what makes "which chat" meaningful. Chosen
filter-with-unattributed-fallback (not annotate-only, not hard-filter) so
multi-chat routing works without orphaning UI-launched or fresh-turn children.

## Edge (accepted, documented)

A child spawned in the orchestrator's very first turn of a brand-new chat has no
session id yet -> `parent_session_id` empty -> unattributed (visible to all chats)
until that turn finishes. Uniform across codex/claude (both use the resumed id,
empty on fresh); claude *could* use its minted id but that would diverge the
backends for a marginal case.

## Verification

- 8 new tests, each written to pin one hop (env, forward, record, filter). A/B:
  removing the env injection reddens `test_orchestrator_mcp_env_has_session_id`;
  neutering the pending filter reddens `test_pending_filtered_by_parent_session`.
- Back-compat: no parent (UI-launched, fresh turn, or older env) -> unattributed
  -> shows to all chats, exactly the pre-change single-chat flow. 19 existing
  comms/pending/request_input tests still pass; full pytest (448) + ruff + mypy +
  `nix flake check` green.

## Self-reflection

The capture was cleaner than feared: the orchestrator's "current chat" is just the
id the turn already resumes, so no session had to be plumbed separately - only
surfaced into the MCP env. The one real design call (filter vs annotate vs
hard-filter) was worth writing down in the DECISION before coding.
