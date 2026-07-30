# Decision: the config repo is a project; only the switch is a capability

- DATE: 20260729-125035
- STATUS: ACCEPTED
- TASK: 20260729-125035
- TAGS: decision, host, nixos, projects, orchestrator, v0.2.0

## Context

`tasks/20260729-125020/DECISION.md` settled the privilege model and left R3 -
the declarative configuration change - to this task. It decided the flow shape
(propose in a worktree over the config repo, commit, build from that commit,
preview with `nix store diff-closures`, activate the exact toplevel) but not who
produces the EDIT. This task's original Steps read as though Scufris would own
that too: "bind the configuration repository as a first-class target", "apply the
change on a branch/worktree of the config repo".

Three shapes were put to the operator, framed as mutually exclusive on their
premise: typed edit verbs where the model may only name a validated nixpkgs
attribute and never emit Nix text; the same with wider verb coverage
(services, firewall ports); or a free-form edit authored by an agent in a
worktree, built and activated as-is.

The operator rejected the frame, and correctly:

> I mean `nix.dotfiles` is "JUST" a project and should be treated as such; the
> only thing that requires privilege is some commands (nixos rebuild switch kind
> of commands) [...] the orchestrator should spawn an agent to modify the
> nix.dotfiles project the same way it would do with any other project.

## Decision

### 1. Scufris owns build, preview, activate and rollback. It does NOT own the edit.

`~/personal/nix.dotfiles` is a flake repo with a git history, an `AGENTS.md` and
a task tree - a project, driven through the machinery every other project uses:
a sprout worktree, a coding agent, a commit, a review. "Add package X" is
plan -> work -> review on that project, and the ONE thing that is not a project
workflow is the activation.

So this task builds the last mile and nothing above it:

- no configuration-repository binding as a bespoke first-class target,
- no typed edit verbs, no anchor-finding in `hosts/nixos/default.nix`,
- no worktree or branch creation by Scufris over the config repo,
- no commit written by Scufris into it.

Rejected: typed edit verbs (`add_package`, `enable_service`, `open_port`). They
would be a second, narrower copy of the project/agent machinery, coupled to one
repo's file layout, covering only what has a verb - and
`tasks/20260729-220835/TASK.md` (the actor-aware orchestrator spike) would then
have to absorb or delete them, since its Projects model already owns "tatr
tasks, lifecycle gates, agent assignments, runs, worktrees, reviews, artifacts".
Its own sentence for the host is "the orchestrator coordinates stage-specific
specialists **and host capabilities**": a capability is the switch, not the
editor.

Consequence for the story's three sentences: "add this package", "open this
port" and "turn on that service" are all answerable, and by the SAME path -
whatever an agent can express in Nix - rather than one verb at a time. The
consequence to accept explicitly is stated in section 3.

### 2. Scufris builds the toplevel from a commit it resolved. A caller may not supply one.

The helper's `activate` verb takes a store path. Where that path comes from is
the load-bearing half, and it is not the caller:

- the app resolves a caller-named REF (branch, tag or sha) in a git repo to a
  concrete rev, and builds
  `git+file://<repo>?ref=<ref>&rev=<rev>#nixosConfigurations.<attr>.config.system.build.toplevel`
  with `--no-link --print-out-paths` and no lock-file writing;
- the generic propose endpoint and the MCP propose tool REFUSE `kind=activate`,
  so no code path exists that activates a path a caller chose.

If the caller supplied the toplevel, the model would pick what gets activated
and the closure diff would faithfully describe whatever it picked. Resolving a
ref to a rev and building it ourselves makes the provenance chain
rev -> build -> toplevel -> approval -> activation, every link recorded, and the
rev is a thing the operator can `git show`.

Building from `?rev=` rather than from a worktree also means Scufris never reads
the working tree, never writes to the repo, and cannot activate an
unreproducible snapshot - the spike's "an identified revision, not a snapshot",
obtained structurally instead of by discipline. Uncommitted files in the agent's
worktree are therefore NOT in the build, which the preview says in as many
words rather than leaving the agent to wonder why its edit did nothing.

### 3. The build runs as the operator, never as root. Stated plainly, so is what that does not protect.

`nixos-rebuild build --flake <ref>` was measured in the spike to run as `alex`
with no root at all, so the helper is not involved in the build - and that is a
security property, not just a convenience. Nix EVALUATION reads files with the
evaluating user's privileges: a configuration evaluated as root can
`builtins.readFile "/root/.ssh/id_ed25519"` or an age key into a derivation
output. As `alex` that read simply fails.

What this does NOT do is make an approved activation safe in the abstract. An
agent-authored configuration can contain `system.activationScripts` or a
`systemd.services.*.ExecStart` that runs anything as root once activated. That
is arbitrary root code reached through a diff, which
`tasks/20260729-125020/DECISION.md` section 5 refused for a SHELL verb - and the
difference is exactly the one the epic is built on and must not be oversold:

- a shell verb has no preview by construction; a configuration change has one
  (the file diff on a reviewed commit, the closure diff, the dry-activate unit
  list), which is why this one is a verb and that one never will be;
- the reviewed commit is the primary control, and it is a PROJECT control (a
  review round on a branch), not a host control;
- the operator approving a closure diff they skim is the residual risk, and it
  is not eliminated by anything in this task.

This paragraph exists so a later reader does not mistake "R3 exists" for "the
model is contained". The containment is that a change must survive a review and
an approval, and that everything it did is in a root-written audit record with
the rev that produced it.

### 4. No `dry_activate` verb. It is part of the activate preview.

`tasks/20260729-125020/DECISION.md` deferred "whether `dry_activate` is worth a
verb in v1". It is not a verb: it is root-only, it changes nothing, and its
output (the units that would restart, the services that would stop) is preview
material. So it runs INSIDE the helper's preview for `activate` and `rollback`,
where it cannot be requested on its own and cannot be mistaken for an action.

A verb whose entire product is preview text would also be the one verb an
approval surface never gates, which is precisely the shape that rots.

### 5. A Plan carries STEPS, not one argv.

Activating a prebuilt toplevel is two privileged commands, in order: set the
system profile to that path, then run its `switch-to-configuration switch`. The
framework landed with one `argv` per plan, so the model grows to a list of
steps, run in order, aborting on the first failure.

Rejected: `nixos-rebuild switch --flake <ref>` as a single verb. It re-evaluates
and rebuilds AS ROOT (section 3), and it would activate the result of a fresh
evaluation rather than the exact store path the operator approved - losing the
one property the whole helper exists for.

The half-applied case is real and is recorded as its own outcome: profile set,
switch failed, which means the NEXT BOOT uses the new configuration while the
running system does not. The audit says that in those words; a failed R3 apply
must never render as "nothing happened".

### 6. Rollback is a targeted activation of a recorded generation.

An applied `activate` records the new generation number and toplevel AND the
ones it replaced, and offers `rollback(generation=<the one it replaced>)` as its
inverse. The helper resolves that generation's toplevel from the system profile
itself; the caller names a NUMBER and never a path.

Rejected: `nixos-rebuild --rollback`, which means "whatever is previous now".
After two changes and a garbage collection, "previous" is a guess, and the
epic's contract is that an undo returns the system to a known generation.

## Consequences

- This task's Steps lose the repository-binding, worktree, edit and commit
  halves. What remains is: steps-in-a-plan, the two R3 verbs with their
  previews and fingerprint, the unprivileged build from a resolved rev, the
  serialization, the API/MCP surface, the audit fields, and the real activation
  proof in the VM test.
- `20260729-125040` (the approval surfaces) renders R3 previews as it renders
  R1/R2 ones: nothing here is a special case for the dashboard.
- The actor-aware orchestrator (`20260729-220835`) can treat a config change as
  a project workflow plus one pending approval, with no host-specific editing
  concept to reconcile.
- `nix.dotfiles` gains nothing new to enable beyond what 20260729-125029
  already required (the sops secret and `services.scufris-hostd.enable`).
- The operator's own checkout is never touched by Scufris at all - not merely
  "not while dirty". Merging an applied change back is a project act
  (`sprout land`), as it already is for every other repo.
