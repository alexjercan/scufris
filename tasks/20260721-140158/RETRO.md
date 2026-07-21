# Retro - 20260721-140158 (Flip dotfiles input to local scufris + rewrite programs.scufris)

## What went well

- The module's flat `settings` -> `SCUFRIS_` mapping made the config rewrite a
  clean, readable swap from the old bot's nested server/bot/telegram/ollama
  block to a short web-server block.
- Building the actual `homeConfigurations.alex.activationPackage` and reading the
  rendered unit gave end-to-end confidence: the alex deployment produces exactly
  the same shape the VM test proved bootable.

## What went wrong / difficulties

- The `allowUnfree` trap: `home/alex/default.nix` already set
  `nixpkgs.config.allowUnfree = true`, but home-configurations.nix passes `pkgs`
  externally, so that setting is IGNORED and codex/claude-code (both unfree)
  would fail eval. Fixed on the pkgs import. Easy to misdiagnose as a module
  bug; it is a home-manager pkgs-provenance issue.
- `path:` copies the untracked working tree wholesale (231M incl. node_modules).
  Expected from the gitignore-independence of `path:`, flagged to the user with
  the `git+file:` alternative rather than silently swapping the input type the
  user asked for.

## Lessons

- `hm-external-pkgs-ignores-nixpkgs-config` (x1): when a home-manager config is
  built with an EXTERNALLY-imported `pkgs` (the common flake-parts pattern:
  `home-manager.lib.homeManagerConfiguration { pkgs = import nixpkgs {...}; }`),
  the in-module `nixpkgs.config.allowUnfree`/overlays are IGNORED. Set
  `config.allowUnfree`/overlays on that external `import nixpkgs {...}`. Symptom:
  an unfree package (codex, claude-code) errors despite an allowUnfree line in
  the user module. 20260721-140158.
- `path-input-copies-untracked-tree` (x1): a `path:/abs/dir` flake input copies
  the ENTIRE directory (no gitignore filter): node_modules/.venv/.git all land
  in the store (here 231M) and re-copy on every content change. Use
  `git+file://<dir>?ref=<branch>` for a git-filtered, reproducible input when the
  copy cost matters; `path:` only when reading the live working tree is the
  point. 20260721-140158.

## Follow-ups

- (user decision, batched to Finish) live `home-manager switch`; optional swap of
  `path:` -> `git+file:`; eventual push/rename and `github:` URL.
