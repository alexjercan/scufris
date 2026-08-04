# Reconcile scufris with the NixOS dotfiles (source of truth, module export, web assets)

- PRIORITY: 20
- TAGS: infra, nix
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Reconcile the locally-developed scufris (`/home/alex/personal/scufris`) with how it
is actually deployed in the NixOS dotfiles, so the code we build here is what runs.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- Finding: `nix.dotfiles` already runs scufris as `systemd.user.services.scufris`
  (home/alex/default.nix) - but via the flake input `github:alexjercan/scufris-bot`,
  NOT this local repo, which exports only `packages`/`devShells` (no
  `homeManagerModules`/`nixosModules`).
- DECISION (user, 20260720): the LOCAL `/home/alex/personal/scufris` is canonical
  and will REPLACE `scufris-bot`. It is not on a remote yet (local only); future
  deployment will be from `/scufris` (renamed, without the `-bot` suffix). This
  repo is the source of truth from here on.
- Work: export `homeManagerModules`/`nixosModules` from THIS flake (it exports only
  `packages`/`devShells` today); ensure the packaged derivation includes the built
  `web/dist` (else the dashboard 404s); then point the dotfiles input at the local
  repo - `path:/home/alex/personal/scufris` in the interim (until pushed to its own
  remote), swapping to the real URL once published. Decide user-service vs
  system-service and whether to expose the web UI (no reverse-proxy precedent in
  the dotfiles today).
- Sequencing: the module export + web-assets-in-derivation work can proceed now
  (local flake work); flipping the dotfiles input over is the last step and may
  wait until the repo is pushed/renamed. Infra task; lower priority.
