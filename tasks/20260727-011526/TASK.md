# Put pkgs.macros on the deployed scufris service PATH

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,nix,deploy,macros

## Goal

Put `pkgs.macros` (and its DB, `~/.local/share/nvim/macros.csv`) reachable on the
DEPLOYED scufris service PATH, so the `macros_*` MCP tools (task 20260727-010447)
work as the running service, not just where `macros` is on the operator's
interactive PATH. Mirrors the `today` deploy (20260726-225845).

## Steps

- [ ] In nix.dotfiles `programs.scufris.path`, append `pkgs.macros` (overlay
      `inputs.macros-nvim.overlays.default` is already applied).
- [ ] Confirm the service can read the macros DB: for the home-manager USER service
      HOME is the real home, so `~/.local/share/nvim/macros.csv` resolves; verify by
      grepping the rendered scufris.service PATH for macros and (optionally) a
      boot/console `macros_lookup` call.

## Notes

- `macros` self-resolves its DB from $HOME; no SCUFRIS_* knob (unlike den_path).
- Batch with the pending `today` PATH change on the same nix.dotfiles branch if not
  yet switched.
