# Retro: Namespace every flake output with a scufris prefix

- TASK: 20260730-164048
- BRANCH: refactor/namespace-flake-outputs
- REVIEW ROUNDS: 2

## What went well

- The gate caught the one real code defect before any human read the diff.
  `nix flake check` failed on `attribute 'web' missing` at
  `nix/scufris-service.nix:63`, which is the one place the rename was
  load-bearing rather than cosmetic. A rename that only touched prose would have
  evaluated fine and shipped a module whose `webPackage` default was dead.
- Deciding the whole output shape at the gate (keep `default`, drop the old
  names, leave `checks.*` alone) meant zero mid-build re-confirmation. The four
  questions were cheap; guessing any of them would not have been.
- Both VM tests were run for real on KVM, twice - by the implementer and again by
  the reviewer at the round-2 tip. For a rename of module ATTRS, evaluation
  proves the attr exists; only the VM test proves it still boots a unit and
  activates a root socket.

## What went wrong

- R1.2 (`.gitignore` kept `nix build .#scufris .#web`). Root cause: the sweep
  enumerated a hand-listed set of surfaces - the docs AGENTS.md names, plus the
  nix and CI files - instead of grepping every tracked file. A tracked dotfile
  carrying a command in a comment IS a surface, and the DoD's own criterion was
  written as "returns nothing" without saying over WHAT. The ledger already had
  this lesson at x1 (`absence-grep-must-not-be-extension-scoped`) and it was not
  applied.
- R1.1 (record CLOSED with no REVIEW.md/RETRO.md, so `records` went red). Root
  cause: `nix flake check` was run, went green, and THEN the task record was
  edited to CLOSED. The `records` check reads that record, so the green result
  described a tree that no longer existed. The gate was treated as a milestone
  already passed rather than as something the next edit can invalidate.
- The step "update the nix sources' own COMMENTS that name outputs" mislabelled
  `nix/scufris-service.nix`. It does name the output in a comment - and also
  resolves it as code (`defaults.web`, reached through
  `self.packages.${pkgs.system}`). The initial grep looked for the dotted literal
  `packages.web`, which the indirection hides.
- Self-inflicted: a script written to re-align a comment column split each line
  on `"#"`, which is also the character in `nix build .#scufris`. It rewrote
  eleven lines of AGENTS.md into nonsense (`nix build .` + `#scufris ...`).
  Caught only because the produced text was re-read, per the global rule that an
  edit is a hypothesis until the artifact shows it.

## What to improve next time

- Prove a rename's absence criterion with `git grep` over every tracked file,
  with PATH exclusions only. Never a hand-listed file set, and say in the DoD
  which corpus the grep covers.
- Run the gate as the LAST action before the commit. Any edit after it - task
  record included, because `checks.records` reads the task record - invalidates
  the green.
- For a nix attribute rename, grep the bare attribute name (`\bweb\b` in
  `nix/`), not just the dotted path: `defaults.web` and
  `self.packages.${system}.web` are the same reference wearing different
  clothes.

## Action items

- [x] ledger: bumped `absence-grep-must-not-be-extension-scoped` to x2, sharpened
      to "`git grep` over all tracked files", since a PATH-scoped grep over a
      hand-listed file set fails the same way an extension-scoped one does.
- [x] ledger: added `rerun-the-gate-after-the-last-record-edit`,
      `a-comments-only-step-can-hide-load-bearing-code`,
      `flake-parts-coerces-nixosmodules-not-homemanagermodules` and
      `dont-split-on-a-char-the-payload-contains`.
- No follow-up code work. The rename is complete and there is no residual
  observation to route.
