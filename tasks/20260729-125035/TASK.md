# Add the NixOS configuration change flow with generation rollback

- PRIORITY: 50
- TAGS: feature, v0.2.0, host, nixos, agents
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want to say "add this package", "open this port", or "turn on
that service" and get a built, diffed, reviewable configuration change, so that
routine NixOS edits stop being a context switch into an editor and a rebuild I
watch scroll by.

This is the payoff task of the epic and the one a general coding CLI cannot do:
it needs the machine, its configuration repository, its privilege, and its
generations, all in one place.

The EDIT, though, is not the part that needs the machine. `~/personal/nix.dotfiles`
is a project like any other, and an agent changes it the way an agent changes any
project: a sprout worktree, a commit, a review (`DECISION.md`, section 1). What
this task builds is the last mile - the part where a reviewed commit becomes the
running system, under an approval, reversibly.

## Steps

- [x] Measure before writing: read the `nixos-rebuild` on this host and confirm
      the exact two privileged commands that activate a prebuilt toplevel, and
      that `nix build 'git+file://<repo>?ref=&rev=#...system.build.toplevel'`
      builds the operator's configuration as `alex` with no lock-file write.
      Record what was measured in `NOTES.md` - the spike measured
      `nixos-rebuild build`, not this argv.
- [x] Sequence the helper's execution: a `Plan` carries STEPS (a list of argvs)
      rather than one argv, because activation is profile-set followed by
      `switch-to-configuration switch`. Mechanical for every existing verb (one
      step each); the preview, the render and the audit name every step.
- [x] Add the R3 verbs to `scufris/hostd/actions.py`: `activate` (a toplevel
      plus the repo/rev provenance) and `rollback` (a generation NUMBER). No
      `dry_activate` verb - it runs inside the previews (`DECISION.md`,
      section 4).
- [x] Validate a toplevel structurally and refuse anything else: a
      `/nix/store/...` path in the charset, no leading dash, a registered valid
      store path, and shaped like a NixOS system (it has
      `bin/switch-to-configuration` and a `nixos-version`). A rollback's
      toplevel is resolved by the helper from the system profile and is never
      supplied by a caller.
- [x] Preview `activate` honestly: `nix store diff-closures /run/current-system
      <toplevel>` with the MEASURED TRAP handled - check the exit status first,
      and render an explicit "no closure change" for empty output on exit 0,
      never a bare empty panel. Label it for what it is.
      SHIPPED WITHOUT the `switch-to-configuration dry-activate` unit list this
      step asked for, deliberately: that binary comes from the toplevel being
      previewed and needs root, so producing the list would run an UNAPPROVED
      configuration's own code as root at propose time. The preview says the list
      is absent and why (REVIEW.md R1.4, NOTES.md).
- [x] Preview `rollback` from the helper's own resolution: the target
      generation's number, date and NixOS version, its toplevel, and the closure
      diff against the running system. No dry-activate here either, for the same
      reason.
- [x] Fingerprint R3 against the running system - the `/run/current-system`
      target and the current generation number - so a system that was switched
      between the preview and the approval refuses as DRIFTED instead of
      applying a stale description.
- [x] Record the inverse: an applied `activate` records the generation and
      toplevel it produced AND the ones it replaced, and offers
      `rollback(<the generation it replaced>)`. A rollback's inverse is the
      generation it left.
- [x] Audit the R3 fields: repo, rev, toplevel, generation before and after, and
      the half-applied outcome (profile set, switch failed - the next boot uses
      the new configuration while the running system does not) as its own
      recorded state rather than a bare failure.
- [x] Build from an identified commit, unprivileged: a new app-side module
      resolves a caller-named ref to a rev in a git repo and builds
      `git+file://<repo>?ref=<ref>&rev=<rev>#nixosConfigurations.<attr>.config.system.build.toplevel`
      as the operator, streamed through the host supervisor and cancellable. It
      never writes to the repo, never reads the working tree into the build, and
      never updates a lock file. The nixosConfiguration attribute defaults to
      this machine's hostname, and an unknown attribute is refused by listing
      the ones that exist.
- [x] Refuse a caller-supplied store path: `POST /api/host/actions` and
      `propose_host_action` reject `kind=activate`, so the only route to an
      activation is a proposal whose toplevel Scufris built from a rev it
      resolved (`DECISION.md`, section 2).
- [x] Surface the change: a route that starts the build and, on success,
      creates the helper proposal; a status read; SSE over the build run; and
      operator-facing text naming the repo, ref, rev and commit subject, whether
      that rev is on the repo's default branch (merging it back is a separate
      project act), and any uncommitted files in the worktree that are therefore
      NOT in this build.
- [x] Serialize: one config build-and-activation at a time per repository,
      refusing rather than interleaving, and reusing the host supervisor's
      existing single-slot behaviour rather than inventing a second one.
- [x] Cover the failure paths as tests: build failure (the log lands on the
      record and no proposal is created), flake lock drift, an unresolvable ref,
      cancellation during the build and during the activation, drift between
      preview and approval, and a second approval of the same change.
- [x] Give the agent the two tools it needs and no more: propose a config change
      from a ref, and read its status. Document in the tool text that editing
      the config is a normal project task and only the switch comes here.
- [x] Prove the half that cannot be faked: extend
      `nix/tests/scufris-hostd-vm.nix` with a REAL activation and rollback
      inside the VM (a specialisation is a real second toplevel in the store,
      which is how NixOS's own switch tests get one).
- [x] Ship `examples/nixos_change.py`: resolve -> build -> preview -> approve ->
      activate -> roll back, end to end against a faked runner and executor.
- [x] Sync the docs in this task: AGENTS.md's privileged-actions section (R3
      exists; what is deliberately NOT Scufris's job), the README if it
      describes the host surface, `CHANGELOG.md`, and the epic's Child Tasks
      rollup.

## Definition of Done

- A change flows resolve -> build -> diff -> approve -> switch with the
  resulting generation recorded, and rolls back
  (test: `test_nixos_change_builds_diffs_switches_and_rolls_back`).
- A build failure surfaces as a failed proposal with its output and never
  reaches activation (test: `test_nixos_build_failure_blocks_activation`).
- The flow never writes to the configuration repository - no worktree, no
  commit, no lock-file update, no working-tree read - so an abandoned or
  rejected proposal cannot have left anything behind
  (test: `test_nixos_change_never_writes_to_the_config_repo`). This replaces the
  planned `test_rejected_nixos_proposal_leaves_repo_clean`: with the edit owned
  by the project flow, cleanliness is structural rather than something a
  teardown has to achieve.
- No caller-chosen store path can be activated: the generic propose surfaces
  refuse `kind=activate`
  (test: `test_propose_refuses_a_caller_supplied_toplevel`).
- Concurrent proposals against the same repository serialize or refuse rather
  than interleaving (test: `test_concurrent_nixos_proposals_are_serialized`).
- A real activation and a real rollback happen in the VM test, as root, on a
  real system profile (cmd: `nix build .#hostd-vm-test`).
- manual: adding a package to the real configuration through chat is faster and
  no scarier than doing it by hand.

## Notes

- Epic: 20260729-124655.
- Depends on: the host action framework (20260729-125029, landed 7677b5f) and
  the host spike's privilege decision (`tasks/20260729-125020/DECISION.md`).
- THE FORK THIS TASK TURNED ON is recorded in `DECISION.md` next to this file:
  the configuration repository is a PROJECT, so Scufris owns build, preview,
  activate and rollback and does not own the edit. The typed-edit-verb shape
  (`add_package` and friends) was rejected there, as was letting a caller hand
  over a toplevel.
- SPIKE OUTCOME - the R3 flow was decided end to end in
  `tasks/20260729-125020/DECISION.md`: build unprivileged, preview with
  `nix store diff-closures`, activate THAT EXACT toplevel and refuse any other,
  roll back to a recorded generation. This task keeps all of that; what it
  changes is who produces the commit being built (an agent working the project,
  not a Scufris edit verb) and how the tree is addressed (`?rev=` on the repo
  rather than a Scufris-created worktree).
- MEASURED TRAP: when the built toplevel matches the running system, `nix store
  diff-closures` exits 0 and prints NOTHING, so "no change" and "the preview
  command failed" are byte-identical in its output. Check the exit status first
  and render an explicit "no closure change"; never show a bare empty panel.
- Rollback records the generation number AND the toplevel at apply time, which
  is what makes rollback a targeted activation instead of a guess.
- Never decrypt, never print, never commit a sops secret. The config repo's
  secrets are sops-nix encrypted in-repo and nothing here reads them; the build
  does not need them.
- Note the reflexive case: this repository is itself a flake input of that
  config, so a Scufris release and a host config change can interact. Keep them
  as separate actions.
- The residual risk this task does NOT remove: an agent-authored configuration
  can run arbitrary code as root once activated, so the reviewed commit and the
  operator's reading of the diff are the controls. `DECISION.md` section 3 says
  this plainly and explains why it is still a verb while a shell verb is not.
