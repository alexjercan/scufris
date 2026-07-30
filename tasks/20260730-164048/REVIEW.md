# Review: Namespace every flake output with a scufris prefix

- TASK: 20260730-164048
- BRANCH: refactor/namespace-flake-outputs

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context

- [x] R1.1 (BLOCKER) tasks/20260730-164048/TASK.md:3 - `nix flake check` FAILS on
  this branch (exit 1). The `records` check reports `closed-missing-review` and
  `closed-missing-retro`: the record is `STATUS: CLOSED` while
  `FLOW STEP: REVIEWING` and the task dir holds only `TASK.md`. The two statuses
  contradict each other and the branch is red on a DoD item TASK.md presents as
  met. Set `STATUS: OPEN` until REVIEW.md and RETRO.md exist, close once they
  land, and re-run `nix flake check`.
  - Response: Confirmed and fixed. `tatr check --ledger LESSONS.md` reproduced
    both errors verbatim. The in-session `nix flake check` had been run BEFORE
    the record was flipped to CLOSED, so it never saw the state the reviewer
    checked out - the gate was green against a tree that no longer existed.
    STATUS is back to `IN_PROGRESS` rather than the `OPEN` the finding asked
    for - the work IS in flight, and `tatr edit --help` lists it as a valid
    status; the reviewer accepted the substitution in round 2. It goes CLOSED in
    the compound step, once RETRO.md is on the branch alongside this file.
- [x] R1.2 (MINOR) .gitignore:24 - stale old output name in a live tracked file:
  `# nix build output symlinks (nix build .#scufris .#web)`. This is a hit for
  the DoD's own sweep regex, so the "no live surface still names an old output"
  criterion does not hold. Change to `.#scufris .#scufris-web`.
  - Response: Confirmed and fixed. The sweep enumerated the doc surfaces
    AGENTS.md names plus the nix and CI files, and never grepped the tracked
    dotfiles - `.gitignore` carries a build command in a comment and so is a
    surface too. Fixed, and the DoD's grep is now run over every tracked file
    rather than a hand-listed set.

Verification notes for this round (not findings):

- The reviewer executed all seven name/default evals, both old-name failure
  probes, `nix build --no-link .#scufris .#scufris-web`, `nix flake check`, and
  BOTH VM tests for real on KVM (`.#scufris-vm-test` and
  `.#scufris-hostd-vm-test`, exit 0 each). So the renamed module attrs are
  proven on a booting unit and a real root socket, not just by evaluation.
- CHANGELOG entries for already-released versions still name `.#web`,
  `.#hostd-vm-test` and `nixosModules.hostd`. Left as written: they describe
  what was true at that version, and the new Unreleased mapping table is what
  translates them. The DoD's grep wording under-specified this.
- `warning: unknown flake output 'homeManagerModules'` from `nix flake check` is
  pre-existing on master and unrelated to this diff.
- No check or test was weakened; the only change under `nix/tests/` is a comment.
- The DoD has no `manual:` items, so there is nothing pending for the user to
  accept beyond the landed diff.

## Round 2

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 reviewer, resumed against the new diff)

Both round-1 findings verified RESOLVED at `ca5f16d`, and no new findings: the
round-1 fixes touched only `.gitignore`, `TASK.md` and this file - nothing in the
flake, the modules, the CI workflows or the tests.

- R1.1: the record reads `STATUS: IN_PROGRESS`, still appears in `tatr ls`, and
  both `nix flake check` and `tatr check --ledger LESSONS.md` exit 0. The
  reviewer accepted `IN_PROGRESS` over the `OPEN` it had asked for as the better
  answer with the same effect.
- R1.2: `.gitignore:24` fixed, and the reviewer ran the sweep itself over every
  tracked file: zero hits outside `tasks/`, `LESSONS.md` and `CHANGELOG.md`, and
  the CHANGELOG hits are exactly the two categories the amended DoD names (the
  Unreleased mapping table, and entries for already-released versions).
- The reviewer re-ran the whole proof set at this tip anyway, including BOTH VM
  tests for real on KVM (exit 0 each). It did not re-run the seven eval probes or
  the two old-name failure probes, on the ground that `flake.nix` is
  byte-identical to the round-1 tip where they all passed.
- RETRO.md is deliberately not on the branch at this point; the record goes
  CLOSED only once it lands, or the `records` check goes red again.
