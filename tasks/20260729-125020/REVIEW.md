# Review: Spike - the host capability privilege and safety model

- TASK: 20260729-125020
- BRANCH: spike/host-privilege-model

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: in-session (exception: this session is configured not to spawn
  subagents, so the out-of-context default was unavailable. Recorded rather
  than silently skipped - a docs-only diff that decides a privilege model is
  substantive by consequence, so this round is weaker than the default and the
  operator may want an out-of-context pass before 20260729-125029 builds on it.)

What was verified rather than taken on trust, since the diff is entirely
claims about how this machine behaves:

- `nix-collect-garbage --dry-run --delete-older-than 3650d` and
  `nix store gc --dry-run` both run unprivileged and report dead paths.
- `nixos-rebuild build --flake ~/personal/nix.dotfiles#nixos` completes as
  `alex` with no root, from the operator's currently DIRTY tree, producing
  `/nix/store/bnfi69...-nixos-system-nixos-26.11...`. The R3 preview chain is
  real, not theoretical.
- `nix store diff-closures /run/current-system ./result` against that build
  runs clean (see R1.5 for what it printed).
- `systemctl list-dependencies --reverse sshd` works unprivileged.
- The middleware claim in DECISION.md section 8 was checked against
  `scufris/app.py` and does not hold - see R1.1.
- `tatr check 20260729-125020`, `tatr check --ledger LESSONS.md` and
  `nix flake check` (ruff, mypy, pytest, records) are all green.

The three DoD `cmd:` proofs were executed and pass. The DoD's `manual:` item
("the operator accepts the privilege model") was cleared by the operator in
session and is recorded on the epic's Manual Acceptance list; it is not
resolved by this verdict.

- [x] R1.1 (MAJOR) tasks/20260729-125020/DECISION.md, section 8 - states "the
  per-process machine token deliberately does not satisfy it, so an agent can
  never approve its own proposal" as though it were an existing property. It is
  not. `scufris/app.py:840-844` short-circuits on a valid bearer token BEFORE
  the session lookup and before the CSRF and same-origin checks, for every
  non-public path and every method. An MCP tool subprocess holds exactly that
  token, so as the code stands today an agent COULD call an approval endpoint.
  This is the most dangerous kind of error in a decision record: a cold reader
  implementing 20260729-125029 would trust the property and omit the check.
  Reword it as a REQUIREMENT on that task - the approval endpoint must reject
  bearer-token authentication explicitly and require a session plus CSRF - and
  state that the current middleware does not provide it.
  - Response: Fixed. Section 8 now states it as a requirement on 20260729-125029, cites `scufris/app.py:840-844` as the reason the property does NOT hold today, and demands a test that a machine-token approval is refused. Carried into that task's Notes as well.
- [x] R1.2 (MAJOR) tasks/20260729-125020/TASK.md:47 - the step "Decide audit
  storage **and retention**" is ticked, but retention is decided nowhere.
  DECISION.md section 7 answers storage (root-owned append-only log, not the
  102147 store) and is silent on how long entries live, how the file is rotated,
  and what bounds its growth - which matters precisely because the log is
  append-only and root-owned, so the app cannot trim it. Either decide it
  (a size or age bound, rotation, and who prunes) or untick the step and carry
  retention explicitly into 20260729-125029.
  - Response: Fixed. DECISION.md section 7 gained a Retention paragraph: helper-owned, size-rotated, bounded rotated set, pruned oldest-first, single-line JSON so rotation never splits a record, and explicitly NO verb for pruning the log. Concrete size/count deferred to 20260729-125029 as a module option, and noted there.
- [x] R1.3 (MINOR) tasks/20260729-125020/DECISION.md:69 - the R2 preview column
  promises "`--dry-run`: bytes freed, generations removed". Measured, it
  delivers neither: `nix-collect-garbage --dry-run --delete-older-than 3650d`
  printed "7642 store paths would be deleted" - a path count, no byte total and
  no generation list. Either narrow the claim to what the tool prints, or say
  the helper computes the size itself (`nix path-info -S` over the dead set)
  and lists the generations. Leaving it as-is repeats in miniature the failure
  this spike calls out elsewhere: presenting adjacent information as a preview.
  - Response: Fixed. The R2 preview cell and section 3 now say what the dry-run actually prints (a dead-path count) and that the helper computes the size itself via `nix path-info -S` and lists the generations it resolved.
- [x] R1.4 (MINOR) tasks/20260729-125020/DECISION.md:87-90 - "a floor that keeps
  the current and the immediately previous generation" is stated as though
  `--delete-older-than <N>d` guarantees it. It does not; that flag keeps the
  current generation and is otherwise purely age-based, so a previous
  generation older than N days is deleted - exactly the R3 rollback target this
  constraint exists to protect. Say that the HELPER enforces the floor by
  resolving the generation list first and refusing or clamping the request,
  rather than implying the flag does it.
  - Response: Fixed. Section 2 now states outright that the flag does not provide the floor and must not be trusted to, and that the helper resolves the generation list first and refuses or clamps a request touching either of the two most recent generations.
- [x] R1.5 (MINOR) tasks/20260729-125020/DECISION.md, section 3 - measured, when
  the built toplevel matches the running system `nix store diff-closures` exits
  0 and prints NOTHING AT ALL. So "no change" and "the diff failed" are
  byte-identical in output, and an approval UI that renders the raw text would
  show an empty panel in both cases. This is a real trap for 20260729-125035:
  note that an empty diff must be rendered as an explicit "no closure change"
  only after the command's exit status has been checked.
  - Response: Fixed. Section 3 records the measured empty-output behaviour as a trap for 20260729-125035, with the exit-status-first rule; also carried into that task's Notes.
- [x] R1.6 (NIT) tasks/20260729-125020/SPIKE.md - no `Fix record` section. The
  spike format reserves it for spikes seeding multiple tasks and this one seeds
  none, so it is defensible - but it refines five children of the epic, and a
  fix record is where a later cycle would look to see which of them have since
  landed. Consider adding the section with the five children listed.
  - Response: Fixed. SPIKE.md gained a Fix record listing the five refined children with their current state.

Pending user checks (not resolved by this verdict):

- Epic 20260729-124655 manual items remain open: the LAN login being bearable
  (from 20260729-125015), the closure diff making a change understandable, and
  the digest being worth reading.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (same exception as round 1)

All six round-1 findings verified resolved against the new diff. Re-checked
the two that mattered:

- R1.1: DECISION.md section 8 now reads as an obligation on 20260729-125029
  with the `scufris/app.py:840-844` short-circuit named as the reason, and the
  same requirement (plus "pin it with a test that a machine-token approval is
  REFUSED") is in that task's Notes, so it survives even if a later reader only
  opens the task.
- R1.2: retention is now decided in section 7 with a shape, not a TODO, and the
  one genuinely deferred part (the concrete size and count) is carried into
  20260729-125029 as a named module option rather than left implicit.

No new findings. `tatr check 20260729-125020`, `tatr check --ledger LESSONS.md`
and `nix flake check` re-run green after the edits.

The DoD's `manual:` item was cleared by the operator in session; the epic's
other manual items stay open for the flow Finish, which an APPROVE does not
resolve.
