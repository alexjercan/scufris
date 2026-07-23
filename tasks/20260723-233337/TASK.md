# Fix test isolation: test_agent_config_omits_builtin_server reads real state dir

- STATUS: OPEN
- PRIORITY: 30
- TAGS: bug,backend,test


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

- [ ] Point the test's `Settings` at an isolated `state_dir=tmp_path` (the
      pattern the other app tests use), so the override store cannot leak in.
- [ ] Grep tests/test_app.py for other `Settings(` constructions that omit
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
