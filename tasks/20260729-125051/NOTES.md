# Notes: continuous integration for every push and pull request

- DATE: 20260729
- TASK: 20260729-125051

## What shipped

`.github/workflows/ci.yaml`, two jobs, both on `ubuntu-latest`:

- **nix** - `DeterminateSystems/nix-installer-action` (pinned by SHA), then
  `nix flake check --print-build-logs`, then `nix build .#scufris .#web`.
- **web** - `actions/setup-node` pinned to node 24.18.0 (the dev shell's
  version), `npm ci`, `npm run ci`.

`flake.nix` gained a `records` check running `tatr check --ledger LESSONS.md`,
with `tatr` as a locked flake input, and `mkCheck` was generalized to
`mkCheckWith tools` so every check shares one sandbox preamble. `tatr` is also
in the dev shell, so a developer runs the version the gate enforces.

## Measured cost (the task's fourth DoD item)

Every CI run is effectively COLD: hosted runners share no Nix store between
runs, so there is no warm case to measure separately. What makes it cheap
anyway is that `cache.nixos.org` supplies the heavy paths as binaries and the
uv2nix set is wheels, so almost nothing compiles.

| Run | Commit | nix job | web job | Result |
|-----|--------|---------|---------|--------|
| 30443720539 | 597eca4 (clean) | 1m51s | 32s | success |
| 30443929343 | bdf1e93 (deliberate break) | 1m33s | 13s | **failure, both jobs** |
| 30444113311 | 7b24f64 (break reverted) | 1m50s | 28s | success |

About two minutes wall clock for a full verdict. That is well inside "does not
discourage small pushes", so the DECISION.md call to skip a third-party binary
cache holds: there is nothing here worth caching yet, and adding Cachix would
buy seconds at the cost of an account and a secret. Revisit if a nixpkgs bump
ever makes something build from source - the 45 minute job timeout exists for
exactly that case.

## Proving it fails (the task's second DoD item)

Run 30443929343 carried a commit that deliberately broke two gates at once: an
unused `import os` at the top of `scufris/health.py` (ruff F401) and a
badly-formatted exported constant appended to `web/src/common.ts` (prettier).
Both jobs went red. The commit was reverted in 7b24f64 and the next run went
green again, so the red was caused by the break and not by something ambient.

The `records` check was proven separately and locally, because corrupting a
task record on a pushed branch would have left a broken record in the history:
setting `FLOW STEP: BANANA` in a task made
`nix build .#checks.x86_64-linux.records` exit 1 with
`bad-flow-state: invalid FLOW STEP 'BANANA'`; restoring the file made it exit
0. That is the same derivation CI builds.

The evidence for the whole workflow lives on PR #1, which exists only to hold
these runs - the branch itself lands through `sprout land`, not through the PR.

## Decisions and why

- **No third-party binary cache.** See `DECISION.md`. Backed by the timings
  above rather than by assumption.
- **Actions pinned by commit SHA.** A tag can be moved; CI is the component
  that is supposed to be trustworthy. Same argument as `flake.lock`, applied to
  the workflow's own dependencies. The human-readable version is in a trailing
  comment so an upgrade is a deliberate edit.
- **`permissions: contents: read`.** This workflow only reads and reports.
- **`cancel-in-progress` on pull requests only.** On master every commit must
  end with its own verdict; cancelling there would let a quick follow-up push
  leave the commit before it with no result.
- **`nix build .#scufris .#web` as a separate step.** `nix flake check` builds
  `checks` but only evaluates `packages`, so without this a stale `npmDepsHash`
  would pass CI while `nix build .#web` was broken for every flake consumer.
- **The NixOS VM test is not here.** It needs KVM and guards the release
  pipeline instead (`tasks/20260729-125101/DECISION.md`). Said so in the
  workflow comments and in AGENTS.md so its absence reads as a decision.

## Still to confirm after landing

Every run recorded above was a `pull_request` run, because PR #1 is where the
evidence was gathered and the branch landed through `sprout land` rather than
through the PR - so the `push: branches: [master]` trigger was unproven at
landing time.

CONFIRMED after landing: run 30445778357 on d531d51 fired with `event: push`
and both jobs green, and run 30446474687 did the same for fc32e42. The push
trigger works. PR #1 was closed unmerged once its evidence had been recorded
here.

## What was harder than expected

Nothing about Nix on the runner - the first cold run was green in under two
minutes, which was the main risk this task carried. The real work was in what
a green check does NOT prove: the first draft ran `nix flake check` and stopped
there, and it took the round-1 review to point out that `packages` are only
evaluated, so the flake could be broken for consumers while CI stayed green.
