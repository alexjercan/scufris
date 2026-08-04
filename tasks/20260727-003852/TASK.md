# Harden den_path Settings test against the operator dev .env (test isolation)

- PRIORITY: 45
- TAGS: bug, test, mcp
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

`test_scufris_mcp_server_injects_den_path_for_orchestrator_only` (test_agent.py)
constructs `_enabled()` = bare `Settings(agent_enabled=True)` for its "unset den"
case and asserts `SCUFRIS_DEN_PATH not in plain.env`. `Settings()` reads the repo
`.env`, so on a dev box whose `.env` sets `SCUFRIS_DEN_PATH` (legit local config to
use the journal tools) the test goes RED, while `nix flake check` (no `.env` in the
sandbox) stays green. This is the `isolate-...-tests-that-assert-config` class
(green on CI, red on a dev box whose override disagrees).

## Steps

- [x] Make the "unset den" assertion hermetic: construct the plain Settings with
      `_env_file=None` (ignore the dev `.env`) so the test proves the code path, not
      the absence of an operator `.env`.
- [x] Grep test_agent.py / test_app.py for other `Settings(...)` assertions that read
      `.env` for a field they assert absent/defaulted; harden any that share the bug.
- [x] Verify: with a `.env` containing `SCUFRIS_DEN_PATH` present, the full suite is
      green (cmd: nix develop -c python -m pytest -q); nix flake check stays green.

## Notes

- Surfaced 2026-07-27 at the Finish of task 20260726-225845 (the operator added
  SCUFRIS_DEN_PATH to their dev .env while testing the journal tools). Bump the
  ledger lesson `isolate-state_dir-in-tests-that-assert-config` to x2.
