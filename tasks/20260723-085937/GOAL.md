# Goal: persisted agent-session registry (fix orchestrator/sub-agent session mixing)

- DATE: 20260723
- UMBRELLA TASK: 20260723-085937
- LANDING SCOPE: squash-merge to local `master` via `sprout land`; no push.

## Goal

The orchestrator's codex/claude session id is held only in memory
(`agent_store.py` `_orch_session_id`), while sub-agents persist theirs on
`AgentRecord`. Both share one `CODEX_HOME`, so after a restart (or wherever a
read path resolves "the session" loosely) the orchestrator can latch onto a
sub-agent's rollout and the conversations mix. This run makes the
`(agent_id -> session_id [+ backend])` mapping a first-class persisted
structure for ALL agents, orchestrator included, and routes every session
resolve/write through it - never through "latest rollout" or a guess. Side
benefit: the orchestrator's conversation survives a restart.

## Done means

1. An orchestrator turn and a codex sub-agent turn keep distinct
   sessions/transcripts, including across a simulated restart. (test:
   reproduction of the mixing, red before the fix)
2. The orchestrator's session id is persisted and restored across a restart
   (no more in-memory-only `_orch_session_id`). (test: restart round-trip)
3. Deleting an agent removes its session mapping. (test)
4. Switching an agent's backend clears the stale cross-backend id instead of
   resuming a wrong-backend session. (test)

Overall: ruff + pytest green; changed source files add zero mypy errors (flake
check mypy leg pre-existing-red, task 20260720-174021).

## Tasks

Updated as tasks land (one line per land, like a spike's Fix record).

- [ ] 20260723-001251 (p42, scufris) Persisted agent<->session id registry
      (fix orchestrator/sub-agent session mixing)

## Decisions (load-bearing, architectural)

- 20260723-001251 DECISION.md: one SessionRegistry file (sessions.json) owns
  every agent's session id, backend-tagged; agents.json stops persisting
  session_id (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

(none yet)

## Notes

- The work task's Notes name two open bugs that may be subsumed:
  20260721-152034 (stale cross-backend session; claude resume fails) and
  20260720-020345 (list app_server sessions / originator fix). Their
  disposition (closed as subsumed, or left open with a note) is decided at
  Finish, not silently dropped.
