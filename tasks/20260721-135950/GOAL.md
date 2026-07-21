# Goal: reconcile scufris with the NixOS dotfiles (modules + web assets)

- DATE: 20260721
- UMBRELLA TASK: 20260721-135950
- ORIGIN TASK: 20260720-122519 (the reconcile task from the spike; closed with this umbrella at Finish)
- LANDING SCOPE: NO push, NO land to master. scufris-side work stays on the
  local branch `infra/nix-dotfiles-reconcile` (master is actively moving and
  must not be touched). nix.dotfiles changes are made directly in
  `/home/alex/personal/nix.dotfiles` (its own repo, committed locally, not
  pushed). This deviates from flow's usual "squash-land each task to the
  default branch" because neither repo may be pushed and scufris master is
  off-limits; per-task work is committed onto the branch instead of landed.

## Goal

Replace the old `scufris-bot` deployment in the NixOS dotfiles with the new
local `scufris` web server (`/home/alex/personal/scufris`,
github.com/alexjercan/scufris). The dotfiles currently run scufris as
`systemd.user.services.scufris` via the flake input `github:alexjercan/scufris-bot`,
whose flake exports a `homeManagerModules.default`. The new local scufris flake
exports only `packages`/`devShells`. This run makes the LOCAL repo the source
of truth: it exports home-manager and nixos modules from the scufris flake,
packages the built `web/dist` dashboard into the nix closure (today's wheel
excludes it, so a packaged run would 404 the dashboard - lesson
`web_dist-via-__file__-is-dev-only`), and points the dotfiles input at the
local repo (`path:/home/alex/personal/scufris` in the interim) with a rewritten
`programs.scufris` for the new web-server interface. The agent backends
(codex/claude) are operator-installed binaries on the service PATH, never
Python deps (lessons `codex-binary-breaks-uv2nix-venv`,
`codex-exec-is-the-nixos-path`).

Depth for this run: plumbing + build-level verification. Not a live run - we
do not start the service and curl the dashboard; we prove it builds and
evaluates.

## Done means

1. The scufris flake exports `homeManagerModules.default` and
   `nixosModules.default`.
   (cmd: from scufris on the branch, `nix eval .#homeManagerModules.default --apply builtins.isAttrs`
   and `nix eval .#nixosModules.default --apply builtins.isAttrs` both print `true`)
2. The scufris flake builds the web dashboard as a derivation whose output
   contains `index.html` and the JS bundles.
   (cmd: `nix build .#web` then `test -f result/index.html`)
3. The home-manager module wires the web assets so a served instance finds the
   dashboard - the generated systemd user unit sets `SCUFRIS_WEB_DIST` to the
   web derivation (or the package bundles it).
   (cmd: build the HM activation and grep the scufris unit for `SCUFRIS_WEB_DIST=/nix/store/...`)
4. `nix flake check` passes on the scufris branch (the existing ruff/mypy/pytest
   gate still green, plus any new module eval).
   (cmd: `nix flake check` in scufris on the branch)
5. The nix.dotfiles `scufris` input points at the local repo
   (`path:/home/alex/personal/scufris`) and `home/alex` uses a rewritten
   `programs.scufris` for the web server; the home-manager configuration
   evaluates and builds.
   (cmd: `nix build` / `home-manager build` of the alex home configuration in
   nix.dotfiles succeeds; `nix flake check` evaluates)
6. Nothing is pushed; scufris master is untouched (still at its pre-run tip);
   scufris work is on `infra/nix-dotfiles-reconcile`.
   (cmd: `git -C ~/personal/scufris log master..infra/nix-dotfiles-reconcile --oneline` shows only this run's commits; master tip unchanged)

Overall: `nix flake check` is green in scufris (branch) AND the nix.dotfiles
alex home configuration builds against the local scufris input.

## Tasks

Updated as tasks complete (one line each; "committed <sha>" replaces "landed"
since nothing lands to master this run).

- [ ] 20260721-140156 (p10, scufris) Package web/dist as a Nix derivation (packages.web)
- [ ] 20260721-140157 (p11, scufris) Export homeManagerModules + nixosModules for the web server
- [ ] 20260721-140158 (p12, nix.dotfiles) Flip dotfiles input to local scufris + rewrite programs.scufris

## Manual acceptance (batched for the user at Finish)

- (pending) whole goal: user decides when/whether to push the scufris branch,
  rename the repo, and swap the interim `path:` input for the real remote URL.
- (pending) whole goal: user runs the live service (`scufris serve`) and
  confirms the dashboard renders, once they choose to activate the new config.
