# Retro: pkgs.macros on the deployed scufris service PATH

- TASK: 20260727-011526
- BRANCH: (nix.dotfiles) master commit 4ae78d2 (on top of the merged today deploy)
- REVIEW ROUNDS: 1 (in-session, trivial; APPROVE)

## What went well

- Reused the exact proof recipe from the today deploy: build the HM activation
  package, grep the rendered scufris.service PATH. Both today-0.1.0 and macros-0.1.0
  present. Mechanical and fast.

## What went wrong

- The nix.dotfiles checkout moved UNDER the flow: the operator concurrently merged
  the today deploy branch into master and switched the checkout to master, so my
  edit+commit (planned for the feature branch) landed on master as 4ae78d2. Net
  result is correct (master now has today+den+macros), but I committed to a default
  branch without re-checking HEAD right before committing. Root cause: I read the
  branch state once at the start and trusted it through the whole cycle in a repo the
  operator was actively editing.

## What to improve next time

- In a repo the user may be touching concurrently (their personal dotfiles),
  re-check `git branch --show-current` / HEAD IMMEDIATELY before committing, not just
  once at the start - the checkout can be switched or merged under you.

## Action items

- [x] Delivered as nix.dotfiles master commit 4ae78d2 (on top of the operator's merge
  of the today+den deploy). Not yet `home-manager switch`-ed - the operator's call.
- [x] Ledger: added `recheck-head-before-committing-in-a-user-touched-repo`.
