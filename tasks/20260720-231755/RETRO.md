# Retro: fix nix flake check pytest derivation

## What went well

- The root cause was exactly the `nix-devshell-import-resolves-to-cwd-source`
  lesson: the mkCheck sandbox ran bare `pytest`, which does not put cwd on
  sys.path, so `import scufris` failed. One-line fix (`python -m pytest`) took it
  from 0 to 224 passing.
- Fixing the first issue surfaced two more sandbox-incompatibilities, and I
  chased each to root rather than stopping at "green enough":
  1. 8 test_agent tests execced fake `codex`/appserver scripts with
     `#!/usr/bin/env bash|python3` shebangs; the nix sandbox has no /usr/bin/env.
     Made the fake-script writers resolve the interpreter (`shutil.which('bash')`,
     `sys.executable`) - a genuine portability fix, so they RUN in the sandbox.
  2. 7 tests shell out to the real `tatr` binary (absent in the sandbox). Gated
     them `@pytest.mark.needs_tatr` + a conftest hook that skips only when tatr is
     off PATH - so they run fully in the devShell and skip loudly in CI.
- Marked precisely: the reject-validation tests (which validate in Python before
  any subprocess) were left running; only the tests that actually invoke tatr are
  gated. Reviewer confirmed the split (including a same-named near-twin pair).

## What went wrong

- The filed task was scoped as "pytest can't import scufris" but the real work
  was 3 distinct sandbox issues (import path, script shebangs, external binary).
  Each needed a different fix; the import one was just the first domino.
- The tatr-in-CI question was a real design fork (bundle vs skip) - surfaced it to
  the user, who chose skip-when-absent (no cross-repo flake coupling).
- Master moved twice during the cycle (parallel agent-orchestrator work); merged
  and re-ran `nix flake check` green before landing each time.

## What to improve next time

- A "make the check green" task can hide multiple independent failures behind the
  first one. Get the FULL failure list early (I read the drv log to enumerate all
  15) rather than fixing the first error and re-running blindly.

## Action items

- [x] nix flake check green (232 pass + 7 skip); devShell 239 pass; landed 8191af7.
- The `python -m pytest` fix + the shebang portability are also candidates to
  note in AGENTS.md if either recurs.
