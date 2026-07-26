# Put today CLI + SCUFRIS_DEN_PATH on the deployed scufris service PATH

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,nix,deploy,journal

## Goal

Put the `today` CLI on the DEPLOYED scufris service PATH so the orchestrator's
`journal_*` MCP tools work off a dev box. Today they only work where `today`
happens to be on PATH (the operator's nix profile); the nixos/home-manager
service module does not add it.

## Context

Landed in tasks/20260720-122514 (the-den journal MCP tools). The MCP server
shells out to `today` via `_run`, which resolves it on PATH. The service module
already puts codex/claude/git on the service PATH (see LESSONS
`scufris-web-server-module-is-env-driven`); `today` belongs in the same set.

## Steps

- [ ] Add the `today` package to the scufris service PATH in the nixos module
      and the home-manager service (wherever codex/claude/git are added).
- [ ] Wire `SCUFRIS_DEN_PATH` from the module's settings (the dotfiles already
      set `user.journal.den_path`) so a deployed box points at the real den.
- [ ] Prove it: the journal tools resolve `today` and read/write the den from
      the deployed service (a VM test or a manual boot-and-call).

## Notes

- `today` is an external repo (~/personal/today) with its own flake; the module
  will need it as an input/overlay or from the same channel that provides it.
