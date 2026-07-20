# Retro: worktree pytest guard

## What went well

- The guard is a fail-fast collection error (RuntimeError in conftest) with a
  message that names both the resolved package root and cwd and points at
  `python -m pytest` - it diagnoses the exact import-shadowing trap.
- Chose the conftest guard over a test wrapper: a wrapper can be bypassed by
  invoking pytest directly, whereas the conftest runs no matter how pytest is
  launched.

## What went wrong

- The environment does NOT reproduce the original failure (bare `pytest` in this
  worktree still imports scufris from the worktree, because REPO_ROOT is set
  per-shell to the worktree and the editable install honors it). So I could not
  demonstrate the bug via bare pytest; I demonstrated the guard by violating its
  invariant directly (running from a foreign cwd), which proves the mechanism.
- First version had a reversed condition that false-fired when running from a
  repo subdirectory. The out-of-context reviewer caught it as a NIT; I fixed it
  to `_pkg_root != _cwd and _pkg_root not in _cwd.parents` (OK when pkg root is
  cwd or an ancestor of it) and re-verified root/subdir/foreign-cwd.

## What to improve next time

- When a guard encodes a directory invariant, enumerate the cwd cases up front
  (root, subdirectory, foreign, symlinked) and test each - the subdirectory case
  is the easy one to get backwards.

## Action items

- [x] Guard landed 19a48f4; R1.1 subdirectory false-fire fixed in-branch.
