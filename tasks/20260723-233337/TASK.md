# Fix test isolation: test_agent_config_omits_builtin_server reads real state dir

- PRIORITY: 30
- TAGS: bug, backend, test
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

`test_agent_config_omits_builtin_server_when_tools_disabled` (tests/test_app.py)
constructs `Settings(web_dist=..., agent_enabled=True, agent_tools_enabled=False)`
WITHOUT setting `state_dir`, so it falls back to the real
`~/.local/state/scufris`. When a developer's settings-override store there has
`agent_tools_enabled: true` in the active profile, the override wins and
`/api/agent/config` returns `tools_enabled=True`, failing the assertion
`body["tools_enabled"] is False`. The test passes only on a machine with an
empty/absent state dir - an isolation bug, not a product bug.

Discovered during task 20260723-225616 (flow umbrella 20260723-225437): the
failure is pre-existing on master and independent of that change.

## Steps

- [x] Point the test's `Settings` at an isolated `state_dir=tmp_path` (the
      pattern the other app tests use), so the override store cannot leak in.
- [x] Grep tests/test_app.py for other `Settings(` constructions that omit
      `state_dir` yet assert on config/tools/profiles, and isolate those too.

## Definition of Done

- `test_agent_config_omits_builtin_server_when_tools_disabled` passes regardless
  of `~/.local/state/scufris` contents (test: the test itself, run with a
  populated real state dir).
- No app test reads the real state dir for a config/tools assertion
  (cmd: `grep -n "Settings(" tests/test_app.py`).

## Notes

- Root cause confirmed: `~/.local/state/scufris/settings.json` default profile
  had `agent_tools_enabled: true`.
- Relevant: `scufris/settings_store.py` (the override store), the app tests'
  `_settings(...)` helper already isolates state_dir - use it / mirror it.

## Outcome (CLOSED)

Added `state_dir=tmp_path` to the two config-asserting tests in tests/test_app.py
that constructed `Settings(...)` without isolating the state dir and thus read the
real `~/.local/state/scufris`:
- `test_agent_config_omits_builtin_server_when_tools_disabled` (was FAILING: the
  real override store had `agent_tools_enabled: true`, so `/api/agent/config`
  returned `tools_enabled=True` and the `assert ... is False` blew up).
- `test_agent_config_reports_effective_settings` (was passing only by COINCIDENCE
  - its asserted backend/tools happened to match the real override; fragile).

Reproduce/verify (A/B): before the fix the first test failed with `assert True is
False` (real state dir populated with `agent_tools_enabled: true`, still untouched
by this change); after adding `state_dir=tmp_path` the FULL suite is green. Since
the real override store still contains `agent_tools_enabled: true`, the now-green
test proves isolation holds regardless of real state-dir contents - the DoD.

Sweep result (DoD `grep -n "Settings(" tests/test_app.py`): every OTHER config/
tools/profiles-asserting test already isolates state_dir - the config-family tests
from L609 on pass `state_dir=tmp_path` (often via `agent_backend=Backend.MOCK`),
and the profiles tests use the `_mock_settings` helper (state_dir=tmp_path).
`test_api_config_exposes_poll_interval` hits `/api/config` (poll_seconds/
agent_enabled), NOT the override-managed `/api/agent/config`, so it is not
fragile and was left as-is. Only the two above needed isolation.

Difficulty: none - the diagnosis was already done when the task was filed (during
umbrella 20260723-225437). The only judgment was scoping the sweep to the
override-managed endpoint (`/api/agent/config`) rather than blindly isolating
every `Settings(` in the file.

Self-reflection: a more root-cause fix would be a conftest autouse fixture that
points state_dir at a tmp for the whole suite, removing this fragility class
entirely - considered but out of scope for a targeted bug fix, and it risks
masking a test that legitimately wants a real state dir. The targeted isolation
matches the task's Steps; if this recurs, promote to the conftest fixture.
