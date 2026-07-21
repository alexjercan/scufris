# Flip dotfiles input to local scufris + rewrite programs.scufris

- STATUS: CLOSED
- PRIORITY: 13
- TAGS: infra, nix

## Story

With the scufris flake exporting modules + web assets (tasks 1-2), point the
nix.dotfiles `scufris` input at the local repo and rewrite `home/alex` to use
the new web-server `programs.scufris`. Today the input is
`github:alexjercan/scufris-bot` and `home/alex/default.nix` configures the old
bot interface (`server.enable`, `bot.enable`, telegram/ollama settings). The
new interface is a single web server. This is the interim wiring
(`path:/home/alex/personal/scufris`), to be swapped for the real remote URL
once the repo is pushed/renamed (that swap is deferred to the user, per GOAL).

## Steps

- [ ] In `nix.dotfiles/flake.nix`, change the `scufris` input from
      `github:alexjercan/scufris-bot` to `path:/home/alex/personal/scufris`
      (keep `inputs.nixpkgs.follows = "nixpkgs"` if compatible; verify the
      local flake's nixpkgs pin does not break the follows).
- [ ] Update `nix.dotfiles/home/alex/default.nix`: rewrite `programs.scufris`
      from the old bot options to the new web-server options (host/port/state/
      agent + environmentFile), keeping the import
      `inputs.scufris.homeManagerModules.default`.
- [ ] Regenerate `nix.dotfiles/flake.lock` for the new input (`nix flake lock`).
- [ ] Verify the alex home configuration evaluates and builds against the local
      scufris.

## Definition of Done

- `nix flake check` in nix.dotfiles evaluates (cmd).
- The alex home configuration builds: `nix build` of the home-manager
  activation package (or `home-manager build`) succeeds and the resulting
  scufris user unit points at the local scufris package + web derivation (cmd).
- The old scufris-bot input is gone; `git -C ~/personal/nix.dotfiles grep scufris-bot`
  finds nothing (cmd).
- Nothing pushed; both repos' changes are local commits only.
- manual: user later swaps `path:` for the real remote URL and pushes when ready.
