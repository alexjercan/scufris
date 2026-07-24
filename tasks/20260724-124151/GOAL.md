# Goal: backend-agnostic orchestrator session endpoints

- DATE: 20260724
- UMBRELLA TASK: 20260724-124151
- LANDING SCOPE: squash-merge to master (local, no push), as in parts 1-2.

## Goal

Part 1 made the orchestrator's session LIST backend-agnostic, but the endpoints
around it still hardcode `resolve_codex_home` + codex module readers, so on a
claude or opencode orchestrator: switching into a listed session shows an empty
transcript, the context readout is wrong/empty, fork reads the codex home, and
delete only unlinks a codex rollout. Make all four route through the
orchestrator's actual backend, mirroring how the per-agent endpoints
(`/api/agents/{id}/transcript`, `/fork`) already use `get_backend(agent.backend)`.

Two new `AgentBackend` protocol methods carry the capabilities the module-level
codex functions provided, so `app.py` never branches on backend:
- `read_context(settings, session_id) -> SessionContext | None` (codex: the rich
  rollout reader; claude/opencode: mapped from `read_status`, window 0 when the
  backend does not report one).
- `delete_session(settings, session_id) -> bool` (codex: unlink rollout; claude:
  unlink the `<id>.jsonl`; opencode: `DELETE /session/{id}`; mock: no-op).

## Done means

1. `get_session_transcript` re-renders a session via the orchestrator's backend
   (test: a claude-backed orchestrator returns the claude session's messages -
   `test_orchestrator_transcript_uses_backend`).
2. `get_context` returns a `SessionContext` via the backend (test: claude-backed
   orchestrator - `test_orchestrator_context_uses_backend`; codex still rich -
   existing `test_context_endpoint_returns_snapshot`).
3. `fork_session` seeds from the backend's transcript (test:
   `test_orchestrator_fork_uses_backend_transcript`).
4. `delete_agent_session` calls the backend's provider delete AND forgets from the
   registry (test: claude delete unlinks the jsonl -
   `test_orchestrator_delete_uses_backend`; codex still unlinks the rollout -
   existing `test_delete_session_removes_and_resets_current`).
5. No `resolve_codex_home` / codex module reader remains in the four orchestrator
   session endpoints (cmd: `grep -n "resolve_codex_home" scufris/app.py` shows
   only the codex-specific usage/memory/account endpoints, not the session ones).

Overall: `nix flake check` green (ruff + mypy + pytest).

## Tasks

- [x] 20260724-124236 (p42, scufris) Route orchestrator session endpoints through the backend
      landed 37ed88c; 1 review round (out-of-context APPROVE, zero findings);
      four endpoints route through the backend, new read_context (sync) +
      delete_session (async) on the AgentBackend protocol.

## Decisions (load-bearing, architectural)

- 20260724-124236 DECISION.md: carry session capabilities on the AgentBackend
  protocol - `read_context` (sync), `delete_session` (async, on the opencode
  client boundary) (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (none expected; all proofs are test:/cmd:)
