# Package web/dist as a Nix derivation (packages.web)

- PRIORITY: 10
- TAGS: infra, nix
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

The scufris FastAPI server serves the built dashboard from `settings.web_dist`
(default `<repo>/web/dist`, derived from `__file__`). This works for the
editable dev install but NOT for a packaged wheel: `web/dist` is gitignored and
the wheel is built with `only-include = ["scufris"]`, so the built derivation
has no frontend and the dashboard 404s (lesson `web_dist-via-__file__-is-dev-only`).
The frontend is a webpack + Tailwind build under `web/` (`web/package.json`,
`package-lock.json`, `node_modules`). We need the built `web/dist` available as
a reproducible Nix derivation so the module can point `SCUFRIS_WEB_DIST` at it.

## Steps

- [x] Inspect `web/package.json` build scripts (the webpack build that produces
      `web/dist`) and `package-lock.json`; confirm the exact build command and
      output dir.
- [x] Add a `web` package to the scufris flake's `perSystem.packages` built with
      `pkgs.buildNpmPackage` (npmDepsHash from `package-lock.json`), running the
      webpack production build and installing `web/dist` to `$out`.
- [x] Handle the gitignored `web/dist`: the flake source is git-filtered, so the
      derivation must BUILD dist from `web/src`, not copy a stale on-disk dist.
- [x] Expose it as `packages.web` (keep `packages.scufris`/`default` unchanged).

## Definition of Done

- `nix build .#web` succeeds and `result/index.html` plus the JS bundles exist
  (cmd: `nix build .#web && test -f result/index.html && ls result`).
- The build is hermetic (no reliance from the on-disk `web/dist`); a clean
  checkout builds it.
- `nix flake check` still passes (existing ruff/mypy/pytest gate untouched).
