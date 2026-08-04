# Test isolation: _ensure_den_path leaks SCUFRIS_DEN_PATH into os.environ across tests

- PRIORITY: 2
- TAGS: backend, tests
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Problem

`tests/test_backends.py::test_claude_stream_args_wires_scufris_for_orchestrator`
fails in the FULL suite (but passes alone) on any checkout whose `.env` sets
`SCUFRIS_DEN_PATH` (the operator's main checkout does; a sprout worktree does
not, which is why it hid during the MCP/Profiles removal task 20260727-123342).

Root cause: `_ensure_den_path` (scufris/app.py:2149) does
`os.environ.setdefault("SCUFRIS_DEN_PATH", str(settings.den_path))` to bridge the
den path to the in-process den probe / MCP subprocess. When an app-creating test
runs with `settings.den_path` populated from the ambient `.env`, this writes
`SCUFRIS_DEN_PATH` into the process environment and NEVER removes it. A later
test that builds `Settings(_env_file=None)` (e.g. `_hermetic()` in
test_backends.py) still reads `os.environ`, so `den_path` is set, `den` gets
wired, and the "no den configured -> scufris only" assertion fails:

    assert allowed == ["mcp__scufris__*"]
    # actual: ["mcp__scufris__*", "mcp__den__*"]

This is a pre-existing latent isolation gap, orthogonal to the MCP/Profiles
removal (test_backends.py was byte-identical before and after that change).

Relates to the promoted `isolate-state_dir` autouse fixture and the
`isolate-config-tests-from-the-ambient-dotenv` /
`settings-test-must-disable-env-file` lessons - `_env_file=None` disables the
`.env` FILE but not leaked `os.environ` vars.

## Definition of Done

1. The full backend suite is green on a checkout whose `.env` sets
   `SCUFRIS_DEN_PATH`. (cmd: `nix develop -c python -m pytest`)
2. The fix isolates the env, not just this one test - a general
   conftest autouse fixture that snapshots/restores `SCUFRIS_*` os.environ keys
   (or pops `SCUFRIS_DEN_PATH`) around each test, so no app-creating test can
   leak into a later env-reading one. Extend the existing autouse
   `_isolate_state_dir` fixture rather than patching each test.
3. A regression check: running `test_app` den tests immediately before
   `test_backends::test_claude_stream_args_wires_scufris_for_orchestrator` stays
   green.

## Steps

- [x] Reproduce: confirm `test_app.py` then the failing `test_backends` test is
      red on a checkout whose `.env` sets `SCUFRIS_DEN_PATH` (done during
      understanding; regression pin below makes it permanent).
- [x] Extend the autouse `_isolate_state_dir` fixture in `tests/conftest.py` to
      a yield fixture that snapshots every `SCUFRIS_*` os.environ key before the
      test and restores that exact set after, so a test writing one directly
      (via `os.environ.setdefault`, which monkeypatch does not track) cannot
      leak into a later env-reading test.
- [x] Add a regression test that pins the isolation: assert that after an app is
      created with a den-configured `Settings`, `SCUFRIS_DEN_PATH` does not
      remain in `os.environ` for a subsequent test (a small test that sets the
      var and relies on the fixture to clear it, or an ordering assertion).
- [x] Verify the full backend suite and full suite are green.

## Notes

- Do NOT change `_ensure_den_path`'s `setdefault` behavior lightly - it is
  load-bearing for the real den MCP subprocess. The fix belongs in test
  isolation (conftest), not production code.
