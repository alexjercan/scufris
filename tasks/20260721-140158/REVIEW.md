# Review - 20260721-140158 (Flip dotfiles input to local scufris + rewrite programs.scufris)

## Round 1 (inline critical pass; reviewer had full context)

Changes in `/home/alex/personal/nix.dotfiles` (local commits, not pushed):
- `flake.nix`: `scufris` input `github:alexjercan/scufris-bot` -> `path:/home/alex/personal/scufris` (keeps `inputs.nixpkgs.follows`).
- `flake/home-configurations.nix`: `config.allowUnfree = true` on the pkgs import.
- `home/alex/default.nix`: `programs.scufris` rewritten for the web server.
- `flake.lock`: relocked (scufris-bot gone; local scufris + its transitive inputs pinned).

### Verified

- `nix flake check --no-build`: all checks pass (homeConfigurations, nixosConfigurations, packages, modules).
- `nix build .#homeConfigurations.alex.activationPackage`: SUCCEEDS. The
  generated `~/.config/systemd/user/scufris.service` has the full SCUFRIS_ env
  set (host/port/log_level + agent_enabled/backend/model/auth_mode),
  `SCUFRIS_WEB_DIST=/nix/store/...scufris-web...`, PATH with codex+claude-code+git+profile,
  `EnvironmentFile=~/.config/scufris/env`, `ExecStart=.../bin/scufris serve`.
- `grep scufris-bot` finds only the explanatory comment - no functional ref.

### Findings

- [minor, ACCEPTED - flagged to user] `path:/home/alex/personal/scufris` copies
  the WHOLE working tree (no git filter): 231M in the store, including
  web/node_modules, on each content change. Chosen deliberately per the task's
  explicit "path: in the interim" and because it reads the working tree (local
  scufris edits apply without a commit). The cleaner alternative, if the copy
  cost bites, is `git+file:///home/alex/personal/scufris?ref=<branch>` (git-
  filtered, ~few MB, reproducible) - a one-line swap; it reads committed state.
  The eventual push/rename swaps this for the real `github:` URL either way.
- [necessary] `allowUnfree` on the home-config pkgs import: codex and claude-code
  are both unfree, and the externally-passed pkgs ignored the in-module
  `nixpkgs.config.allowUnfree`. Set on the import to honor the user's already-
  declared intent. Blast radius: all home configs may now use unfree pkgs -
  matches what `home/alex` already declared.
- [scope] Live activation (`home-manager switch`) is the user's call and is
  batched as a Finish manual-acceptance item; the VM test (20260721-141458)
  already proves the service boots and serves.

- VERDICT: APPROVE
