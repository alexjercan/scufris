# Reconcile scufris with the NixOS dotfiles (source of truth, module export, web assets)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: infra,nix

## Goal

Reconcile the locally-developed scufris (`/home/alex/personal/scufris`) with how it
is actually deployed in the NixOS dotfiles, so the code we build here is what runs.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- Finding: `nix.dotfiles` already runs scufris as `systemd.user.services.scufris`
  (home/alex/default.nix) - but via the flake input `github:alexjercan/scufris-bot`,
  NOT this local repo, which exports only `packages`/`devShells` (no
  `homeManagerModules`/`nixosModules`).
- OPEN QUESTION (user decision, blocks this): which repo is canonical - local or
  the GitHub scufris-bot? 
- If local becomes canonical: export `homeManagerModules`/`nixosModules` from this
  flake and point the dotfiles input at `path:/home/alex/personal/scufris`; ensure
  the packaged derivation includes the built `web/dist` (else the dashboard 404s);
  decide user-service vs system-service and whether to expose the web UI (no
  reverse-proxy precedent in the dotfiles today).
- Infra task; lower priority; do not start before the source-of-truth decision.
