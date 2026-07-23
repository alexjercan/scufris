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

- [x] 20260723-001251 (p42, scufris) Persisted agent<->session id registry
      (fix orchestrator/sub-agent session mixing)
      landed b877782; 1 review round (APPROVE, 1 MINOR fixed in-cycle: mark_finished
      keys the session by the run's launch-time backend). 340 tests green.

## Decisions (load-bearing, architectural)

- 20260723-001251 DECISION.md: one SessionRegistry file (sessions.json) owns
  every agent's session id, backend-tagged; agents.json stops persisting
  session_id (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

(none - all four done-definition items have `test:` proofs; no manual checks.)

## Finish (2026-07-23)

Done-definition verified item by item on master (commit d163283):
1. Orchestrator + codex sub-agent stay distinct across a restart -
   `test_orchestrator_and_subagent_sessions_stay_distinct_across_restart` PASSED.
2. Orchestrator session persisted/restored across restart - same test's restart
   leg PASSED.
3. Deleting an agent removes its mapping - `test_delete_removes_session_mapping`
   PASSED.
4. Backend switch clears the stale cross-backend id -
   `test_backend_switch_clears_session_mapping` PASSED (plus the legacy-migration
   and R1.1 launch-backend tests).

Overall green bar: full suite 340 passed, `ruff check .` clean, `mypy` clean on
the touched files, `tatr check --ledger LESSONS.md` exit 0.

Post-land incident: the landed b877782 had the R1.1 fix reverted by a
sabotage-test's `git checkout --` (the fix was not yet committed), so app.py
called a `mark_finished(backend=...)` whose param was gone - the persist
callback raised and sessions were not persisted. Caught by THIS Finish gate (5
test_app failures on master), fixed in d163283. Recorded as ledger lesson
`commit-before-sabotage-or-the-restore-eats-the-fix` and in the task RETRO.

Sibling-bug disposition: 20260721-152034 (stale cross-backend session) and
20260720-020345 (originator/list) are BOTH already CLOSED - nothing to subsume;
the registry complements their prior fixes rather than replacing them. No
deferred items, no unresolved findings, no dropped tasks.
