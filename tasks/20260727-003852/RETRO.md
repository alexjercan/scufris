# Retro: harden den_path Settings test against the dev .env

- TASK: 20260727-003852
- BRANCH: (none - hotfix committed on master; test-only, source-of-truth gate stayed green)
- REVIEW ROUNDS: 0 (trivial test-isolation diff; A/B: red with the dev .env before, 524 green after)

## What went well

- The flow Finish full-suite run caught it: `nix flake check` (no `.env` in the
  sandbox) was green, but `nix develop` with the operator's `.env` was red. Running
  the suite at Finish, not just the source-of-truth gate, surfaced a real dev-box
  papercut.
- The fix targeted the shared `_enabled()` helper (`_env_file=None`), so it hardens
  every `_enabled()`-based absence assertion (den_path AND the latent disabled_tools
  one), not just the one failing line.

## What went wrong

- The previous task's test (`test_scufris_mcp_server_injects_den_path_for_orchestrator_only`)
  used a bare `Settings()` as its "unset den" baseline, which reads the repo `.env`.
  Root cause: the `isolate-state_dir-in-tests-that-assert-config` lesson was applied
  to the STATE store but not generalized to the `.env` file - the same class, second
  occurrence. The lesson even predicted it ("red on a dev box whose override
  disagrees").

## What to improve next time

- A Settings-asserting test's baseline must be hermetic against BOTH external config
  sources - the state override store (`state_dir=tmp_path`) and the `.env`
  (`_env_file=None`). At the next recurrence, promote to a conftest autouse
  hermetic-Settings fixture (bumped the ledger toward that).

## Action items

- [x] `_enabled()` made hermetic (`_env_file=None`); suite green with the dev `.env`
  present (524 passed).
- [x] Ledger: bumped `isolate-state_dir-in-tests-that-assert-config` to x2 with the
  `.env` variant.
