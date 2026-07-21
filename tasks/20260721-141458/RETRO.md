# Retro - 20260721-141458 (NixOS VM test for the scufris service)

## What went well

- Adapting the old scufris-bot VM test was fast: the module already resolves
  its own packages from `self.packages.${system}`, so the test only passes
  `scufrisModule` (no `scufrisPackage`), simpler than the original.
- The test is a genuine end-to-end proof: it caught nothing broken because T1
  and T2 were already solid, but it converts the plumbing-only run's "no live
  run" gap into a real boot-and-serve check that lives in the repo and reruns
  on demand. It exercises T2's DynamicUser state-dir fix for real.
- Asserting on `GET /` content (`Scufris`, `id="app"`) pins the exact failure
  the web packaging exists to prevent (API-only 404 dashboard).

## What went wrong / difficulties

- The new `.nix` file was untracked, so the first `nix build .#vm-test` failed
  with "not tracked by Git" - a dirty-tree flake sees modified tracked files but
  NOT new untracked ones. The outer `... ; echo EXIT=$?` also masked the real
  non-zero exit (the echo always succeeds) - had to grep the log for the true
  result. Lesson reinforced: run the build bare or capture `$?` of the build
  itself, not a trailing echo.

## Lessons

- `flake-cant-see-untracked-new-files` (x1): a dirty-tree flake evaluation
  includes modifications to TRACKED files but not brand-new untracked files;
  `nix build` fails with "Path ... is not tracked by Git". `git add` the new
  file (explicit path, never `-A` in this repo) before building. 20260721-141458.
- `nixos-vm-test-for-on-demand-not-checks` (x1): expose a `pkgs.testers.nixosTest`
  as `packages.vm-test` (Linux-only via `lib.optionalAttrs pkgs.stdenv.isLinux`),
  not a `checks` entry, so the fast lint/type/test gate is not dragged down by a
  full VM boot; run it deliberately with `nix build .#vm-test`. 20260721-141458.
