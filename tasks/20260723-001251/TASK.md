# Persisted agent<->session id registry (fix orchestrator/sub-agent session mixing)

- STATUS: CLOSED
- PRIORITY: 42
- TAGS: bug,agents,backend,sessions
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the operator, I want each agent's (including the orchestrator's) codex/claude
session tracked by an explicit, persisted `(agent_id -> session_id)` mapping, so the
orchestrator's conversation never gets mixed up with a sub-agent's when both run on
codex.

## Context (grounded / root cause)

The orchestrator's session id is IN-MEMORY only (`agent_store.py:112`,
`self._orch_session_id`, set/read via `set_orchestrator_session` /
`orchestrator_session_id`) and is NOT persisted - it resets on restart. Sub-agents
persist `session_id` on their `AgentRecord` (`agents.json`). Both share one
`CODEX_HOME`. When a read path (transcript/status) resolves "the session" loosely,
or the orchestrator's in-memory id is `None` after a restart, it can latch onto the
most-recent codex rollout - which may be a sub-agent's. Same-backend (both codex)
makes them indistinguishable to a loose resolver, which is the mixing observed.

## Direction

Make the session<->agent mapping a first-class, persisted structure for ALL agents
(orchestrator included), and resolve sessions ONLY through it. Mechanism chosen in
DECISION.md (this folder): a `SessionRegistry` JSON file
(`<state_dir>/sessions.json`, `agent_id -> {backend, session_id}`) owned by
`AgentStore`; `agents.json` stops persisting `session_id`.

## Steps

- [x] Reproduce FIRST (red): in `tests/test_agent_store.py`, record an
      orchestrator session and a codex sub-agent session via the run-state
      mutators (`mark_finished` / `set_orchestrator_session`), then rebuild the
      store over the same `state_dir` (simulated restart) and assert the
      orchestrator's id survives and stays distinct from the sub-agent's. Watch
      it fail: today `orchestrator_session_id()` comes back None after rebuild.
- [x] Add `SessionRegistry` in `scufris/agent_store.py`: persisted
      `<state_dir>/sessions.json`, atomic write + tolerant load (mirror
      `AgentStore._load`/`_persist`). API: `get(agent_id, backend) -> str | None`
      (None on backend mismatch), `set(agent_id, backend, session_id)`,
      `clear(agent_id)`.
- [x] Route `AgentStore` through it: construct the registry in `__init__`; delete
      `_orch_session_id`; `set_orchestrator_session` / `orchestrator_session_id`
      delegate to the registry under `canonical_backend(settings.agent_backend)`;
      `mark_finished` writes the captured session id to the registry for every
      agent id (orchestrator branch included).
- [x] Populate `AgentRecord.session_id` from the registry at read time (`get`,
      `list`, `_orchestrator_record`) keyed by the record's current backend; stop
      persisting `session_id` in `_persist`; migrate a legacy `session_id` found
      in `agents.json` at `_load` into the registry (only when the registry has
      no entry for that agent yet).
- [x] `delete(agent_id)` removes the registry mapping; `update(...)` on a backend
      switch clears it (replacing the in-record `updates["session_id"] = None`
      dance - the backend key already makes the stale id unreachable, but clear
      it so sessions.json does not accumulate dead entries).
- [x] Update existing tests that assert `session_id` round-trips through
      `agents.json` to the registry semantics; add delete-clears-mapping,
      backend-switch-clears and legacy-migration tests. (As executed: NO
      existing test needed changing - the registry preserves the whole
      observable AgentStore API, so test_agent_store/test_app/test_mcp_server
      passed unmodified; only the four new tests were added.)
- [x] Verify: `ruff format` + `ruff check .` + `python -m pytest` from the
      worktree; mypy on the touched files adds no new errors.

## Definition of Done

- Orchestrator and a codex sub-agent never share a session/transcript, including
  across a simulated restart. (test:
  `test_orchestrator_and_subagent_sessions_stay_distinct_across_restart`)
- The orchestrator's session id is persisted and restored across a restart.
  (test: same reproduction test, the restart leg)
- Deleting an agent removes its session mapping. (test:
  `test_delete_removes_session_mapping`)
- Switching an agent's backend clears the stale id instead of resuming a
  wrong-backend session. (test: `test_backend_switch_clears_session_mapping`)
- A legacy `agents.json` `session_id` migrates into the registry on load. (test:
  `test_legacy_agents_json_session_id_migrates_to_registry`)

## Notes

- Two `AgentStore` instances exist in production (app.py:575 and
  mcp_server.py:174) over the same state files; sessions.json inherits the same
  last-write-wins semantics agents.json already has - accepted, not widened.
- The turn path (`app.py` `_launch_agent_turn`) already resolves strictly by
  `agent.session_id`; no "latest rollout" resolver exists in the backend read
  paths (`read_status`/`read_transcript` return None/[] on a None id). The
  loose-latch risk is the restart data loss plus the cross-backend stale id -
  both closed by the registry.

- Folds/relates to open bugs: `20260721-152034` (stale cross-backend session; claude
  resume fails) and `20260720-020345` (list app_server sessions / originator fix) -
  consider subsuming or closing them.
- Lesson context (`LESSONS.md`): `backends-tag-provenance-differently` (codex exec
  vs app-server originator); `probe-the-stateful-path-not-the-one-shot`.
- Side benefit: persisting the orchestrator's session survives a restart (today its
  conversation is lost).

## Close record (2026-07-23)

What changed: added `SessionRegistry` to `scufris/agent_store.py` - a persisted
`<state_dir>/sessions.json` mapping `agent_id -> {backend, session_id}`, atomic
write + tolerant load like the sibling stores. `AgentStore` now routes ALL
session ids through it: the in-memory `_orch_session_id` field is gone
(`set_orchestrator_session`/`orchestrator_session_id` delegate under the
current settings backend), `mark_finished` writes every captured id to the
registry, `agents.json` no longer persists `session_id` (excluded in
`_persist`; a legacy value migrates into the registry once at `_load`),
`get`/`list`/`update`/`mark_*` attach the id at read time via `_with_session`,
`delete` clears the mapping, and a backend switch clears it instead of nulling
the record field. Internal records keep `session_id = None` (a `_raw` accessor
keeps mutators from leaking a registry-attached copy back into the dict).
CHANGELOG Fixed entry added.

Evidence: the reproduction
(`test_orchestrator_and_subagent_sessions_stay_distinct_across_restart`) was
written first and failed red on the pre-fix code with
`AssertionError: assert None == 'orch-sess'` (the restart losing the
orchestrator's in-memory id - exactly the reported mechanism), and went green
with the registry. Full suite: 339 tests, all green (exit 0) (`python -m pytest` from the
worktree), `ruff check .` clean, `mypy scufris/agent_store.py` clean.

Alternatives considered: recorded in DECISION.md (sidecar-for-orchestrator-only
and per-record backend-tagged dict rejected for keeping two storage shapes).

Difficulties: none of the existing tests needed changing - the observable API
is unchanged, which was the design intent but also means the old
"session_id round-trips via agents.json" behavior was never directly pinned.
One process slip: an early edit landed in the MAIN checkout instead of the
worktree (the Edit tool followed the file read before sprouting); caught by
`git status` on the main tree and reverted before any commit.

Self-reflection: read files from the WORKTREE path from the start (re-read
after `sprout new`, never edit a path read pre-sprout). The repro test could
additionally drive the app-level `/api/chat` path; kept at store level because
the supervisor's persist path funnels into `mark_finished` either way.

Note on scope: the sibling bugs named in Notes (20260721-152034 cross-backend
clear, 20260720-020345 originator/list) were NOT closed here; disposition is
decided at the flow Finish step.

## Review round 1 follow-up (2026-07-23)

Out-of-context review APPROVEd (REVIEW.md R1). Addressed R1.1 (MINOR): a
sub-agent backend switch via `update_agent` is NOT serialized against an
in-flight turn (app.py:983 `update_agent` has no `supervisor.serialized`
guard), so a switch could land mid-run and `mark_finished` would then re-record
the just-finished codex session under the new claude label - defeating the
registry's backend-mismatch guard. Fix: `mark_finished` takes an optional
`backend` (the backend the run executed under); the supervisor persist callback
passes the launch-time snapshot's `agent.backend` (app.py:1116). Pinned by
`test_mark_finished_keys_session_by_run_backend_not_current`, sabotage-verified
to fail (`assert 'codex-sess-late' is None`) without the fix. Suite now 340
green. R1.2 (NIT, per-record persist in the legacy migration) left as-is:
harmless at real agent counts, noted for if the code is touched again.
