# Retro: today + SCUFRIS_DEN_PATH on the deployed scufris service

- TASK: 20260726-225845
- BRANCH: (nix.dotfiles) feature/scufris-today-den-path
- REVIEW ROUNDS: 1 (in-session, trivial diff; APPROVE)

See TASK.md for the verified module facts and the scope decision; process only here.

## What went well

- Understand-first stopped a mis-scoped build. The task text (inherited from the
  parent task's follow-up note) said to change the "nixos module and home-manager
  service" and add a "VM test". Reading the module first showed it ALREADY supports
  both knobs generically (`path` is operator-populated; `den_path` flows through the
  `settings` -> `SCUFRIS_*` map), so the real fix was two lines of operator config in
  nix.dotfiles, not scufris code. Surfacing that as a named fork (AskUserQuestion,
  constraint stated) let the user pick the nix.dotfiles-only scope instead of me
  building a VM test and a flake-input coupling that were not needed.
- Proved it against the real artifact, not an eval: built the HM activation package
  and grepped the rendered `scufris.service` for the PATH entry and the env var
  (lesson `render-hm-unit-file-not-eval`), rather than trusting a nix eval.

## What went wrong

- Nothing costly. The one latent trap avoided: `home.sessionVariables.DEN_PATH` is
  already set for the interactive shell, which could have looked like enough - but a
  systemd USER service does not inherit HM session vars, so the service would have
  had no den path without the explicit `settings.den_path`. Caught during
  understanding by reasoning about the systemd env, and pinned by grepping the
  rendered unit.

## What to improve next time

- When a follow-up task's text prescribes a mechanism ("add a VM test", "change the
  module"), re-derive the mechanism from the current code before accepting it - a
  note written at parent-task time can prescribe work the module already does.

## Action items

- [x] Delivered in nix.dotfiles (branch feature/scufris-today-den-path); left
  unmerged/unswitched for the user (machine-affecting).
- [x] Ledger: added `systemd-user-service-ignores-hm-session-vars`.
