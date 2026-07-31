# Split the agent runtime modules under the size cap

- STATUS: OPEN
- PRIORITY: 90
- TAGS: refactor, v0.2.0, agents, backend, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the agent runtime split along its real seams, so that
changing one backend, one store, or the session index does not require loading
four thousand lines of unrelated agent code.

## Steps

- [ ] Characterize current behavior with the existing suites before moving
      code; add integration coverage for any seam that has none.
- [ ] Split `scufris/backends.py` (1098) by backend adapter and shared
      protocol; the shared contract stays one module.
- [ ] Split `scufris/agent_store.py` (1035) by persistence concern versus run
      lifecycle concern.
- [ ] Split `scufris/sessions.py` (835) into ownership index and session
      records/queries.
- [ ] Split `scufris/agent.py` (832) along orchestrator turn versus transport
      and event plumbing.
- [ ] Keep public import paths stable, or update every caller in the same
      change. No compatibility shim modules.
- [ ] Apply the epic comment policy to every file touched.
- [ ] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- Every file under `scufris/` touched by this task is at or under 600 lines
  and the allowlist no longer names them (cmd: `python scripts/check_file_size.py`).
- No public behavior or import contract drifts
  (cmd: `python -m pytest`).
- Backend selection and health remain identical for codex, claude, opencode,
  and mock (test: `test_backends.py`).
- Session ownership and multi-session history unchanged
  (test: `test_sessions.py`).
- Agent run lifecycle, resume, cancel, and outcomes unchanged
  (test: `test_agent_store.py`, `test_agent.py`).
- `scufris/README.md` module map matches the new layout
  (cmd: `rg -n "backends|agent_store|sessions" scufris/README.md`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420 (guard and comment policy must exist first).
- Split along ownership, not line count. A mechanical split that keeps the same
  coupling is a failure.
- Do not fold in behavior fixes; file a task instead.
