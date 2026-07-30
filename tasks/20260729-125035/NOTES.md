# Notes: the NixOS configuration change flow (R3)

What shipped, what was measured, and what changed course mid-build. The FORK this
task turned on is `DECISION.md` next to this file; this is the implementation
record.

## What shipped

Helper side (`scufris/hostd/`):

- `actions.py`: `RiskClass.R3`, the `activate` and `rollback` verbs, structural
  toplevel validation, and `Plan.steps` - a plan is now a SEQUENCE of `Step`s
  (argv + label + timeout) rather than one argv, because activating a prebuilt
  system is two privileged commands in order.
- `nixos.py` (new): the R3 previews (closure diff, generation resolution), the
  R3 fingerprint (running store path + generation number), and the
  switch-in-flight preflight.
- `files.py` (new): the `Files` seam - `is_file`, `is_executable`, `resolve` -
  because R3 asks questions of the store that no command answers honestly, and
  shelling out to `test`/`readlink` would add a failure mode (the unit's PATH
  holds `nix`, `systemd` and `nixos-rebuild`, deliberately not coreutils).
- `engine.py`: runs steps in order, stops at the first failure, records how far
  it got, and applies a per-class apply-time preflight distinct from the drift
  check.

App side:

- `hostconfig.py` (new): resolve a ref to a rev with real git, build
  `git+file://<repo>?ref=&rev=#nixosConfigurations.<attr>...toplevel`
  unprivileged and streamed, then propose the activation of what was built. A
  build failure is terminal and carries its log.
- `app.py`: `POST /api/host/config/changes` (+ list, get, SSE, cancel), and the
  refusal of `kind=activate` on the generic propose surface.
- `mcp_server.py`: `propose_nixos_change` and `nixos_change_status`; `rollback`
  reachable through the existing propose tool.

Proofs: `tests/test_nixos_config_change.py` (34 tests, including the four the
Definition of Done names), `examples/nixos_change.py` (four modes), and a REAL
activation plus rollback as root in `nix/tests/scufris-hostd-vm.nix`.

## Measured on this host, before writing the code

- `nix build 'git+file:///home/alex/personal/nix.dotfiles?ref=master&rev=<sha>#nixosConfigurations.nixos.config.system.build.toplevel'
  --no-link --print-out-paths` runs as `alex` with no root, took 6.4s with a warm
  eval cache, and returned exactly the store path the machine was already
  running - which also proved the running system was built from that rev. The
  repository stayed clean: no `result` symlink, no lock-file write.
- `nix store diff-closures A A` exits 0 and prints NOTHING (the trap the spike
  recorded, re-confirmed).
- `nix store diff-closures` emits ANSI colour codes and non-ASCII glyphs
  (`->`, the empty set) even into a pipe, and `NO_COLOR=1` does NOT suppress
  them. A real generation-to-generation diff was 398 lines. So the preview
  strips escapes, transliterates the glyphs and bounds the list at 60 lines.
- `<toplevel>/bin/switch-to-configuration dry-activate` as `alex`:
  "Error: switch-to-configuration must be run as the root user", exit 1.
- The real activation commands, read out of this host's `nixos-rebuild` (a
  python `nixos-rebuild-ng`, `nixos_rebuild/nix.py`): `nix-env -p <profile>
  --set <toplevel>`, guarded by a check that `<toplevel>/nixos-version` exists,
  then `systemd-run --collect --no-ask-password --pipe --quiet
  --service-type=exec --unit=nixos-rebuild-switch-to-configuration --
  <toplevel>/bin/switch-to-configuration switch`. Both are copied rather than
  invented, including the unit name.

## Decisions taken during the build

**No `dry_activate`, anywhere.** The plan said the preview would include the
unit-restart list from `switch-to-configuration dry-activate`. Writing it made
the problem obvious: that binary comes FROM the toplevel being previewed, which
is a configuration an agent wrote and nobody has approved, and running it needs
root. A preview that executes unapproved code as root would make "proposing
changes nothing" depend on the proposed configuration behaving well. So the
preview shows the closure diff and says in words why the unit list is absent.
This also answers the question `tasks/20260729-125020/DECISION.md` deferred
("is `dry_activate` worth a verb in v1"): it is worth neither a verb nor a place
in the preview. The operator was told at hand-back rather than after the fact.

**`Plan.argv` became `Plan.steps`, a clean break.** The audit record's `argv`
field became `steps`, so a record written by an older build shows no command
(pydantic ignores the unknown key). That was acceptable precisely once: hostd has
never been enabled on the operator's machine - the sops secret and
`services.scufris-hostd.enable` are still pending - so no production audit log
exists to become less readable. A later rename would not have this luxury.

**One build per repository is REFUSED, not queued.** The supervisor could
serialize them, but a queued NixOS build sits for an hour with no visible reason,
and two builds of one repository contend for the same evaluation anyway. The
`serialize_key` is still set as a backstop for the race.

**Configuration builds get their own supervisor.** Sharing the host-apply
supervisor's single slot would have let a kernel rebuild block an unrelated
approved service restart. Builds are unprivileged; applies are not.

## Bugs and surprises, and how they were found

- **`nix path-info` needs `--extra-experimental-features nix-command`.** Found by
  the VM test, not by reading: with a default `nix.conf` the new CLI is disabled
  outright, so the helper refused a perfectly valid store path with a reason that
  had nothing to do with the store. It also means the ALREADY-SHIPPED R2 verbs
  (`nix store gc`, and `nix path-info --json` in the dead-set preview) were
  broken on any host that had not opted in - this host has, which is why nobody
  noticed. Fixed at the source: `host.run.nix_cli` puts the features in every new-CLI
  argv, the way `nixos-rebuild` does.
- **A NixOS test VM has NO system profile generation.** `nix-env -p
  /nix/var/nix/profiles/system --list-generations` is empty there, because the VM
  boots its toplevel directly - while every installed host has at least one. The
  test now creates the generation an installed machine already has, and says why.
  The code was already correct for the empty case (the reversal reports
  `possible=False` rather than guessing a target), which is how the failure
  surfaced as an assertion in the test rather than a crash in the helper.
- **The rollback's switch installs a bootloader, and a test VM cannot.** The
  forward activation passed; the rollback reached `install-grub.pl` and failed
  with "will not proceed with blocklists" on the test image's ext2 root. That is
  the environment, not the code - and the code reported it exactly right:
  `step 2 of 2 failed after 1 succeeded` plus "THIS boot still runs the old one
  while the NEXT boot would run the new one". The split-state path was proved by
  accident before it was asserted on purpose. `boot.loader.grub.enable = false`
  in the test node removes the environment's limitation.
- **`git status --porcelain` output was being sliced by column** after being
  `.strip()`ed, so the leading space of an unstaged ` M path` was already gone
  and every reported filename lost its first character. Caught by a test using a
  REAL git repository; a faked git would have agreed with the bug.
- **A fake store path has to be a plausible one.** The first fixture hashes
  contained `e`, `t` and `u`, which are not in nix's base-32 alphabet, so the
  validator refused them - correctly. Fixture data that cannot exist tests
  nothing (LESSONS.md, `a-fixture-that-cannot-express-the-bug-blesses-it`).

## What could have gone better

- The three plan steps about editing the config repo (bind it, sprout a
  worktree, commit) were written before the operator pointed out that
  `nix.dotfiles` is just a project. The signal was already in the task's own
  Story - "add this package", "open this port", "turn on that service" are three
  typed verbs, i.e. three narrow surfaces - and in the actor-aware orchestrator
  spike's Projects model. Asking "who owns the edit" is the question that should
  have opened the planning phase, not the artifact-shape question that followed
  it.
- Both VM-test findings were reachable by reading (`nix.conf` features; a test
  VM's profile) but were found by running. The VM test earned its cost in one
  cycle, and the honest lesson is to run it EARLIER - it was the last step in the
  plan and the first thing that found a real bug.
