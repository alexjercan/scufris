# Review: retire the codex-exec runner + fix the settings-view backend picker

- TASK: 20260721-180224
- BRANCH: feature/retire-exec-runner

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings from a fresh subagent with no sight
  of the implementing session; the in-session pass re-ran both suites and
  adopted both NITs, re-deriving the strict-input contract via the new test)

Both check suites pass: backend `ruff` + `mypy` + `pytest` (all green) and web
`npm run ci` (prettier + eslint + vitest + webpack build). The diff delivers the
Goal: the dead codex-exec runners and their orphaned helpers are gone (DoD grep
`_run_codex_exec\|_stream_codex_exec` -> empty), the app-server runner is the sole
survivor with its shared helpers intact, `agent_backend` is widened to the
canonical `codex|claude|mock` (legacy `app_server|exec -> codex` coercion kept for
load, API input strict), health probes the selected backend, and the settings
picker is server-authoritative (Codex/Claude, Mock only behind the dev flag; DoD
grep `app_server` in settings-view.ts -> empty).

- [x] R1.1 (NIT) tests/test_health.py - the `test_agent_health_probes_claude_backend`
  docstring said "a missing claude binary is an error", but the test uses a broken
  (present-but-non-runnable) fake bin, which warns. Reword to match the exercised
  path.
  - Response: Fixed. Docstring now reads "a broken claude binary warns".

- [x] R1.2 (NIT) tests/test_app.py - no test pinned the strict-input contract (a
  raw `app_server` PATCH is rejected 422) though the task calls it load-bearing.
  Add a regression test.
  - Response: Added `test_patch_agent_config_rejects_legacy_backend_id`: PATCH
    `agent_backend="app_server"` -> 422, documenting that new writes must use the
    canonical vocab while env/state still coerces on load. Confirmed green.

### Pending manual DoD (user's to eyeball; APPROVE does not resolve it)

- manual: on the settings page the backend picker shows Codex/Claude (+ Mock only
  in dev), and switching the orchestrator to Claude runs the landing chat on it.
