# Persisted agent<->session id registry (fix orchestrator/sub-agent session mixing)

- STATUS: OPEN
- PRIORITY: 42
- TAGS: bug,agents,backend,sessions

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

- [ ] Reproduce FIRST (red): in `tests/test_agent_store.py`, record an
      orchestrator session and a codex sub-agent session via the run-state
      mutators (`mark_finished` / `set_orchestrator_session`), then rebuild the
      store over the same `state_dir` (simulated restart) and assert the
      orchestrator's id survives and stays distinct from the sub-agent's. Watch
      it fail: today `orchestrator_session_id()` comes back None after rebuild.
- [ ] Add `SessionRegistry` in `scufris/agent_store.py`: persisted
      `<state_dir>/sessions.json`, atomic write + tolerant load (mirror
      `AgentStore._load`/`_persist`). API: `get(agent_id, backend) -> str | None`
      (None on backend mismatch), `set(agent_id, backend, session_id)`,
      `clear(agent_id)`.
- [ ] Route `AgentStore` through it: construct the registry in `__init__`; delete
      `_orch_session_id`; `set_orchestrator_session` / `orchestrator_session_id`
      delegate to the registry under `canonical_backend(settings.agent_backend)`;
      `mark_finished` writes the captured session id to the registry for every
      agent id (orchestrator branch included).
- [ ] Populate `AgentRecord.session_id` from the registry at read time (`get`,
      `list`, `_orchestrator_record`) keyed by the record's current backend; stop
      persisting `session_id` in `_persist`; migrate a legacy `session_id` found
      in `agents.json` at `_load` into the registry (only when the registry has
      no entry for that agent yet).
- [ ] `delete(agent_id)` removes the registry mapping; `update(...)` on a backend
      switch clears it (replacing the in-record `updates["session_id"] = None`
      dance - the backend key already makes the stale id unreachable, but clear
      it so sessions.json does not accumulate dead entries).
- [ ] Update existing tests that assert `session_id` round-trips through
      `agents.json` (test_agent_store, test_app, test_mcp_server) to the registry
      semantics; add delete-clears-mapping, backend-switch-clears and
      legacy-migration tests.
- [ ] Verify: `ruff format` + `ruff check .` + `python -m pytest` from the
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
