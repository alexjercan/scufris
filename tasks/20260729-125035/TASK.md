# Add the NixOS configuration change flow with generation rollback

- STATUS: OPEN
- PRIORITY: 50
- TAGS: feature,v0.2.0,host,nixos,agents

## Story

As the operator, I want to say "add this package", "open this port", or "turn on
that service" and get a built, diffed, reviewable configuration change, so that
routine NixOS edits stop being a context switch into an editor and a rebuild I
watch scroll by.

This is the payoff task of the epic and the one a general coding CLI cannot do:
it needs the machine, its configuration repository, its privilege, and its
generations, all in one place.

## Steps

- [ ] Bind the configuration repository (`~/personal/nix.dotfiles`) as a
      first-class target: located by configuration, validated as a flake, and
      refused politely when dirty or absent.
- [ ] Edit in isolation: apply the change on a branch/worktree of the config
      repo (reuse the existing sprout machinery rather than editing the live
      tree), so an abandoned proposal leaves nothing behind.
- [ ] Build before showing: run `nixos-rebuild build` (or the flake equivalent)
      against the proposed tree and surface build failures as the proposal's
      result rather than as an agent monologue.
- [ ] Diff what will change: file diff plus closure diff (`nvd diff` or
      `nix store diff-closures`) between the running system and the built one,
      rendered as the action preview.
- [ ] Apply on approval: activate the built configuration, record the resulting
      generation number, and capture the activation output.
- [ ] Roll back: one control returns the system to the previous generation and
      records the rollback as its own audited action.
- [ ] Handle the repository side honestly: commit the change with a message
      naming the agent and run, leave pushing to the operator, and never rewrite
      history.
- [ ] Cover the failure paths: build failure, activation failure, dirty repo,
      flake input drift, concurrent proposals against the same repo, and
      cancellation during build or activation.

## Definition of Done

- A change flows edit -> build -> diff -> approve -> switch with the resulting
  generation recorded, and rolls back
  (test: `test_nixos_change_builds_diffs_switches_and_rolls_back`).
- A build failure surfaces as a failed proposal with its output and never
  reaches activation (test: `test_nixos_build_failure_blocks_activation`).
- An abandoned or rejected proposal leaves the configuration repository as it
  was (test: `test_rejected_nixos_proposal_leaves_repo_clean`).
- Concurrent proposals against the same repository serialize or refuse rather
  than interleaving (test: `test_concurrent_nixos_proposals_are_serialized`).
- manual: adding a package to the real configuration through chat is faster and
  no scarier than doing it by hand.

## Notes

- Epic: 20260729-124655.
- Depends on: the host action framework and the host spike's privilege decision
  (settled - `tasks/20260729-125020/DECISION.md`).
- SPIKE OUTCOME - the R3 flow is decided end to end: propose in a SPROUT
  WORKTREE over the config repo, COMMIT on that branch, `nixos-rebuild build
  --flake <that commit>` (unprivileged), preview with `nix store diff-closures
  /run/current-system ./result`, operator approves, then the `scufris-hostd`
  helper activates THAT EXACT toplevel store path and refuses any other. The
  operator's own checkout is never touched (it is dirty today, which is
  precisely why); merging the branch back is a separate operator act.
- Building from a commit rather than a dirty tree is deliberate: what gets
  activated must be an identified revision, not an unreproducible snapshot.
- Rollback records the generation number AND the toplevel path at apply time,
  which is what makes rollback a targeted activation instead of a guess.
- MEASURED TRAP: when the built toplevel matches the running system, `nix store
  diff-closures` exits 0 and prints NOTHING, so "no change" and "the preview
  command failed" are byte-identical in its output. Check the exit status first
  and render an explicit "no closure change"; never show a bare empty panel.
- `nixos-rebuild build --flake <ref>` was verified to run as `alex` with no root
  at all (it built the operator's config straight through), so the whole preview
  half of this flow needs no helper - only the activation does.
- The config repo uses flake-parts with `hosts/nixos` and home-manager under
  `home/alex`; secrets are sops-nix encrypted in-repo. Never decrypt, never
  print, never commit a decrypted secret.
- Note the reflexive case: this repository is itself a flake input of that
  config, so a Scufris release and a host config change can interact. Keep them
  as separate actions.

## Flow State

- FLOW STEP: PLANNING
