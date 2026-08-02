# Put pkgs.macros on the deployed scufris service PATH

- PRIORITY: 30
- TAGS: feature, nix, deploy, macros
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Flow State

- WORKING NOTE: nix.dotfiles branch feature/scufris-today-den-path

## Goal

Put `pkgs.macros` (and its DB, `~/.local/share/nvim/macros.csv`) reachable on the
DEPLOYED scufris service PATH, so the `macros_*` MCP tools (task 20260727-010447)
work as the running service, not just where `macros` is on the operator's
interactive PATH. Mirrors the `today` deploy (20260726-225845).

## Understanding (verified 2026-07-27)

- nix.dotfiles-only change (scufris repo gets only the task record), like the `today`
  deploy. The macros CLI self-resolves its DB from `$HOME` -> no SCUFRIS_* knob.
- The macros overlay is already applied (`inputs.macros-nvim.overlays.default`,
  flake/home-configurations.nix:27), so `pkgs.macros` is in scope.
- The pending `today` deploy branch `feature/scufris-today-den-path` is NOT yet merged
  to nix.dotfiles master or switched. Its path line is already
  `[pkgs.codex pkgs.claude-code pkgs.git pkgs.today]`. Per the task note, BATCH the
  macros append onto that same branch so one merge+switch deploys both.
- HM USER service HOME is the real home, so the running service reads
  `~/.local/share/nvim/macros.csv`.

## Steps

- [x] On the nix.dotfiles branch `feature/scufris-today-den-path`, append `pkgs.macros`
      to `programs.scufris.path` -> `[pkgs.codex pkgs.claude-code pkgs.git pkgs.today
      pkgs.macros]`.
- [x] Prove it: build `homeConfigurations.alex.activationPackage` and grep the
      rendered `scufris.service` `PATH=` for `macros` (and confirm `today` still there).

## Definition of Done

1. The rendered scufris HM unit has `pkgs.macros` on its `PATH=` (alongside today +
   codex/claude/git). (cmd: build the HM config, grep the generated scufris.service)
2. The HM config still builds clean. (cmd: `nix build .#homeConfigurations.alex.activationPackage`)

## Notes

- `macros` self-resolves its DB from $HOME; no SCUFRIS_* knob (unlike den_path).
- Landing: the code lands on the nix.dotfiles branch (batched with today); the scufris
  task record is the only scufris-repo artifact. Merge+switch is the user's call.
