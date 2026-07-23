# Review: Fix test isolation - config tests read real state dir

- TASK: 20260723-233337
- BRANCH: fix/test-state-dir-isolation

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff - two `state_dir=tmp_path` additions to
  test-only code; correctness self-evident and proven by A/B red->green with the
  real state dir still populated, so an out-of-context round adds no signal)

No findings.

Verification: Reproduced the bug on the branch base (the target test failed with
`assert True is False` because the real `~/.local/state/scufris/settings.json`
has `agent_tools_enabled: true`). After the fix the full Python suite is green
(`python -m pytest` exit 0; ruff format/check + mypy clean) while that override
file is UNCHANGED - so the now-green test proves isolation holds regardless of
real state-dir contents (DoD item 1). Re-derived the sweep-completeness claim
(DoD item 2) independently: enumerated every test touching `/api/agent/config` or
`/api/agent/profiles`; all except the two fixed here already isolate `state_dir`
- the config-family tests (L609+) via explicit `state_dir=tmp_path` (often with
`Backend.MOCK`), the profiles tests via the `_mock_settings(tmp_path)` helper.
`test_api_config_exposes_poll_interval` hits `/api/config` (not the
override-managed `/api/agent/config`) and `test_openapi_docs_are_organized`
asserts on OpenAPI structure, not config values - neither is fragile, correctly
left as-is. No assertion was weakened or removed (only `state_dir=tmp_path` was
added). No open `manual:` items.
