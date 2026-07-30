# Review: the NixOS configuration change flow (R3)

- TASK: 20260729-125035
- BRANCH: feat/nixos-config-change
- ROUNDS: 1
- VERDICT: APPROVE

## Round 1

Reviewed IN CONTEXT, not by an out-of-context reviewer, and that is a weakness of
this review rather than a judgement that the diff was small: subagents are
disabled for this session, so the mechanism `/review` prefers was unavailable.
The diff is ~1500 lines of production code across a privileged surface, which is
exactly the shape that benefits most from a reader who did not write it. Recorded
here so the next reader knows what this verdict is worth.

The pass therefore leaned on the two things that do not depend on who is reading:
running the VM test against a real root helper (which found two findings nobody
was going to read their way to), and sabotaging each new guard to prove its test
goes red.

### Findings

**R1.1 (MAJOR, fixed) - the caller chose the repository, so "the server builds
what it activates" was one step from meaningless.** `POST /api/host/config/changes`
took `repo` from the request body and built whatever flake was there.
`DECISION.md` section 2 argues that a caller must not choose the store path,
because then the closure diff faithfully describes the caller's own choice; a
caller-chosen REPOSITORY is the same property one step removed - an agent can
write a flake anywhere it can write, commit it, and have the server build and
propose that as this machine's system. The preview does name the repository and
the revision, so it was visible rather than hidden, but "visible in a prompt" is
the control this epic exists to stop relying on alone.
Fixed: `resolve(..., allowed=settings.host_config_repo)` checks the resolved MAIN
repository, so any WORKTREE of the configured configuration repository passes -
which is exactly where an agent works - and nothing else does. Which revision to
build stays the caller's business.
Test: `test_a_repository_other_than_this_host_s_configuration_is_refused` (both
halves: the foreign repo is refused, a worktree of the real one is not).

**R1.2 (MAJOR, fixed) - the tool an agent always calls would have reported a
timeout for a build that was running.** The route probed
`nixosConfigurations.<attr>` before answering, to turn an unknown host into a
sentence naming the ones that exist. That probe is a full flake EVALUATION -
measured 6.4s warm on this host and slower cold, which is precisely the state
after a configuration change - and `mcp_common._API_TIMEOUT` is 15s. So
`propose_nixos_change` would return "error: timed out" while the change proceeded
in the background: the worst possible answer, because the agent then either
reports failure or starts a second one.
Fixed: the probe moved into the build run, before the build. The request now does
git reads only (milliseconds), and a bad attribute lands on the record as a
failure with the same message.
Test: `test_the_attribute_probe_does_not_delay_the_request` (201, then the change
settles FAILED naming the attributes that exist, with no proposal).

**R1.3 (MINOR, fixed) - `ref: HEAD @ 3af39d5` in an approval prompt tells the
operator nothing.** The default ref was recorded verbatim, so the audit record,
the render and the flake reference all said "HEAD" - the one value that cannot be
looked up later. Fixed: `resolve` translates HEAD to the branch it is (a detached
HEAD keeps the object id).
Test: `test_a_ref_of_head_is_recorded_as_the_branch_it_is`.

**R1.4 (MAJOR, fixed during the build) - the preview ran the proposed
configuration's own code as root.** Planned: include the unit-restart list from
`switch-to-configuration dry-activate`. That binary comes FROM the toplevel being
previewed, needs root, and at propose time nobody has approved anything - so the
framework's first promise ("proposing changes nothing") would have depended on an
unapproved configuration behaving well. Fixed by removing it entirely and saying
so in the preview text; `tasks/20260729-125020/DECISION.md`'s deferred question
("is `dry_activate` worth a verb") is answered "neither a verb nor a preview
line". Surfaced to the operator at hand-back rather than after the fact.
Test: `test_the_preview_never_runs_the_proposed_configuration` asserts no runner
or executor call touches `switch-to-configuration` at propose time, AND that the
preview says why the list is absent.

**R1.5 (MAJOR, fixed) - every `nix` new-CLI call assumed the operator's
`nix.conf`.** Found by the VM test, not by reading: with default nix settings the
new CLI is disabled, so `nix path-info` failed and the helper refused a perfectly
valid store path with a reason that had nothing to do with the store. The
already-shipped R2 verbs had the same latent break (`nix store gc`, and the
dead-set preview's `nix path-info --json`) - invisible on this host because it has
opted in. Fixed at the source: `host.run.nix_cli` puts
`--extra-experimental-features "nix-command flakes"` in every new-CLI argv, the
way `nixos-rebuild` does.

**R1.6 (MINOR, accepted) - a leftover FAILED transient unit would make
`systemd-run` refuse with a systemd error rather than our sentence.**
`switch_in_flight` treats `active`/`activating`/`reloading`/`deactivating` as in
flight, correctly not `failed` (a failed unit is not running). `--collect` should
garbage-collect a failed transient unit, so the window is narrow, and the outcome
is a failed step 2 with systemd's own message in the record - diagnosable, not
silent. Not worth a second code path.

**R1.7 (MINOR, accepted) - the audit's `argv` field became `steps`, so records
written by an older build show no command.** Acceptable exactly once: hostd has
never been enabled on the operator's machine (the sops secret and
`services.scufris-hostd.enable` are still pending), so no production audit exists
to become less readable. Recorded in NOTES.md because a later rename will not
have this luxury.

### Checked and found sound

- The check-and-put for "one build per repository" has no `await` between the
  in-flight check and the store write, so two racing requests cannot both pass
  it. The 409 is the real behaviour, not just the tested one; the
  `serialize_key` remains as a backstop.
- `?ref=X&rev=Y` was tested against real nix: the REV is what gets locked even
  when the ref does not contain it, and `ref=HEAD` resolves. So the build is
  pinned to the revision in every form this code produces.
- Nix flake evaluation is PURE by default, so a configuration cannot
  `builtins.readFile "/home/alex/.ssh/..."` at eval time even though the build
  runs as the operator. The reason the build must not run as root is narrower
  than "eval reads files": it is that root's own eval-time reads would succeed
  where the operator's fail.
- A cancelled build leaves no proposal, and a failed build leaves none either, so
  "nothing to approve" is structural rather than a check.
- The R1/R2 previews and argv are unchanged apart from the mechanical
  `argv` -> `steps[0].argv` move, and their existing tests still assert the same
  commands.

## Verdict

APPROVE. The three findings above are fixed with tests that were each sabotaged
to prove they go red; the full gate (`ruff`, `mypy`, `pytest`, records) is green,
and `nix build .#hostd-vm-test` passes with a real root activation and a real
rollback.
