# Put today CLI + SCUFRIS_DEN_PATH on the deployed scufris service PATH

- PRIORITY: 30
- TAGS: feature, nix, deploy, journal
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Put the `today` CLI on the DEPLOYED scufris service PATH and point it at the-den,
so the orchestrator's `journal_*` MCP tools work as the running service (not just
where `today` happens to be on the operator's interactive PATH).

## Understanding (verified 2026-07-27)

scufris's service module ALREADY supports this fully - nothing to change in the
scufris repo:
- `programs.scufris.path` (nix/scufris-service.nix:107) takes any package list;
  the operator already sets `[pkgs.codex pkgs.claude-code pkgs.git]`. Adding
  `pkgs.today` is an operator edit, not a module change.
- `den_path` is a `config.py` field, so it flows through the GENERIC
  `settings` -> `SCUFRIS_<UPPER>` mapping (nix/scufris-service.nix:43); no module
  code needed. The MCP subprocess reads `SCUFRIS_DEN_PATH` (orchestrator-only
  injection preserves the isolation).
- `today` is published (`github:alexjercan/today`) with a ready `overlays.default`;
  nix.dotfiles already applies it (flake/home-configurations.nix:26) and installs
  `pkgs.today` (home/modules/scripts/default.nix:11), so `pkgs.today` is in scope.
- The service does NOT inherit `home.sessionVariables.DEN_PATH` (systemd user
  services don't see HM session vars), so `SCUFRIS_DEN_PATH` must be set
  EXPLICITLY in `programs.scufris.settings`.
- Only the home-manager `programs.scufris` (home/alex/default.nix:131) is
  configured; there is no NixOS `services.scufris`, so this is HM-only.

DECISION (user, 2026-07-27): scope this as nix.dotfiles-ONLY. The scufris repo
gets no code change; the fix is the operator wiring in nix.dotfiles, proven by a
`home-manager build` + rendered-unit grep. See the AskUserQuestion answer.

## Steps (in ~/personal/nix.dotfiles)

- [x] In `home/alex/default.nix` `programs.scufris.path`, append `pkgs.today`
      to the existing `[pkgs.codex pkgs.claude-code pkgs.git]`.
- [x] In `programs.scufris.settings`, add `den_path = "/home/alex/personal/the-den";`
      (matching the `DEN_PATH` session var in home/modules/scripts/default.nix).
- [x] Prove it: `home-manager build` (or the flake's HM activation package) and
      grep the RENDERED `scufris.service` unit for `today` on `PATH=` and for
      `SCUFRIS_DEN_PATH=/home/alex/personal/the-den`.

## Definition of Done

1. The rendered scufris home-manager unit has `pkgs.today` on its `PATH=` and
   `SCUFRIS_DEN_PATH=/home/alex/personal/the-den` in its `Environment=`.
   (cmd: build the HM config and grep the generated `scufris.service`)
2. The HM config still evaluates/builds clean (no eval error from the edit).
   (cmd: `home-manager build` / `nix build` the activation package)

## Notes

- No scufris-repo code change; the scufris task record (this TASK.md -> CLOSED
  plus a short RETRO) is the only thing that lands in scufris. The wiring commit
  lands in nix.dotfiles.
- `today` in scope via the already-applied `inputs.today.overlays.default`.
