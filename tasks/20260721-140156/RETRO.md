# Retro - 20260721-140156 (Package web/dist as a Nix derivation)

## What went well

- The `buildNpmPackage` fake-hash bootstrap (build once with an all-A sha256,
  read the "got:" hash from the mismatch error, pin it) worked first try.
- Building from `web/src` sidestepped the gitignored-`web/dist` trap cleanly:
  the Nix output is a fresh, more-complete production build than the stale
  on-disk dist, so there is no "which dist am I serving" ambiguity.

## What went wrong / difficulties

- None material. The one thing to watch: `dontNpmInstall = true` plus a custom
  `installPhase` is required because this is a static-asset build, not an
  installable npm library - the default buildNpmPackage install would fail
  looking for a package to pack.

## Lessons

- `buildnpmpackage-static-site-needs-dontNpmInstall` (x1): for a webpack/vite
  app that emits static files (not a publishable package), use
  `dontNpmInstall = true` + a custom `installPhase` copying the build output to
  `$out`; the default install/pack phase has no package to install and fails.
  Pair with `npmBuildScript = "build"`. Resolves the open half of
  `web_dist-via-__file__-is-dev-only`: the closure now CAN carry the built
  assets. 20260721-140156.

## Follow-ups

- Hardcoded web version + shipped source maps are noted in REVIEW.md as
  low-priority cleanups, not filed as tasks (cosmetic).
