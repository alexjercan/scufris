# Review: pkgs.macros on the deployed scufris service PATH

- TASK: 20260727-011526
- BRANCH: (nix.dotfiles) master commit 4ae78d2 (on top of the operator's merge of the today deploy)

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff: one package appended to programs.scufris.path,
  proven by building the real HM activation package and grepping the rendered unit)

Change: `pkgs.macros` appended to `programs.scufris.path` in
`home/alex/default.nix`. NOTE: mid-flow the operator merged the today deploy branch
into nix.dotfiles master and the checkout was on master, so this landed as master
commit 4ae78d2 ON TOP of the merged today+den change (not on the feature branch as
first planned) - the net state is the same: master carries today+den+macros.

Verification (load-bearing claim re-derived, not a finding): built
`homeConfigurations.alex.activationPackage` and read the generated
`scufris.service` (lesson `render-hm-unit-file-not-eval`). Its `PATH=` carries
`...today-0.1.0/bin:...macros-0.1.0/bin:...` (both present; today not regressed),
and `SCUFRIS_DEN_PATH=/home/alex/personal/the-den` still set. The config built
clean. `pkgs.macros` resolves via the already-applied `inputs.macros-nvim.overlays.default`.
`macros` self-resolves its DB from `$HOME`, and the HM USER service runs with the
real home, so `~/.local/share/nvim/macros.csv` resolves at runtime - no knob needed.
Confirmed only `home/alex/default.nix` was staged (pre-existing dirty flake.lock not
swept in).

No BLOCKER/MAJOR/MINOR/NIT findings. No open `manual:` DoD items. APPROVE.
