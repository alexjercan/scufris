# Retro: pre-commit guard for web/node_modules

## What went well

- The guard is versioned (hooks/pre-commit in the tree) and activated via
  core.hooksPath=hooks in the existing devShell shellHook, so it travels with
  the repo and applies in main + every sprout worktree without per-clone setup.
- Verified the guard end to end (staged symlink blocks the commit, HEAD
  unchanged; normal commit succeeds) and the reviewer independently reproduced
  it plus the regex anchoring (web/node_modules_notes.md does not match) and the
  set -euo pipefail + grep-in-if-condition safety.

## What went wrong

- `nix flake check` is red on master before this branch: its mkCheck pytest
  derivation fails with `ModuleNotFoundError: No module named 'scufris'` (the
  sandbox does not install the package). Confirmed as a baseline failure, not a
  regression (baseline-dod-proofs lesson). Verified my change via the devShell
  instead (ruff + mypy + `python -m pytest` = 203 passed).

## What to improve next time

- The broken sandbox pytest check means the goal's "nix flake check green" bar
  can't be met until it's fixed. Filing it as its own task rather than expanding
  this branch.

## Action items

- [x] Guard landed 7fc2ed0.
- Follow-up task to file: fix the mkCheck pytest derivation so `nix flake check`
  installs/imports scufris (sandbox pytest currently red on master).
