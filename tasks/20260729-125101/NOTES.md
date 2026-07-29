# Notes: publish a GitHub Release from a version tag

- DATE: 20260729
- TASK: 20260729-125101

## What shipped

`.github/workflows/release.yaml`, three jobs in a chain so that everything
expensive depends on the cheap checks and nothing is published before the
artifact is proven to run:

1. **guard** - resolves the version from the tag (or the `workflow_dispatch`
   input), classifies anything beyond `MAJOR.MINOR.PATCH` as a pre-release, and
   runs `scripts/check-release-ready.sh`. Both later jobs `need` it, so a
   failing guard blocks the release rather than annotating it.
2. **verify** - re-runs the full gate on the TAGGED commit (`nix flake check`,
   `nix build .#scufris .#web`, the frontend suite) plus the NixOS VM test.
   Re-running rather than trusting master's CI matters because a tag can point
   at any commit, including one that never saw a pull request.
3. **publish** - `uv build`, smoke-test the wheel, extract the notes, create or
   update the release, upload assets with `--clobber`. The only job with
   `contents: write`.

`scripts/check-release-ready.sh` is the guard, and it is the same script an
operator runs before tagging. It checks: version agreement (tag, pyproject,
changelog), that the version has real release notes, `tatr check --ledger
LESSONS.md`, that no uncompiled scratch is sitting in `docs/`, and that the
working tree is clean.

## The VM test: decided by measurement, not by assumption

The DECISION said to attempt the NixOS VM test in the release gate and REMOVE
it if the runner had no KVM. The probe (temporary `kvm-probe.yaml`, deleted in
this task) found something more interesting than either branch:

- `/dev/kvm` EXISTS on `ubuntu-latest`...
- ...as `root:kvm` mode 0660, with the runner user not in the `kvm` group, so
  it is not readable or writable.
- After `sudo chmod 666 /dev/kvm` the device works and
  `nix build .#vm-test` passes in **102 seconds** (run 30446677138).

So the step stays, unconditionally. The reason this is worth writing down: the
obvious guard, `if [ -e /dev/kvm ]`, is WRONG on this runner. It passes while
the device is unusable, and in its skip-form it would let a release publish
having tested nothing - the exact failure the DECISION was written to prevent,
reached by the exact check that looks like it prevents it. The workflow instead
fixes the permission and runs the test outright, so losing KVM makes the
release go red.

## Idempotence

`gh release view || gh release create`, then `gh release edit` on the existing
one, then `gh release upload --clobber`. Re-running the workflow for a tag that
already has a release converges on the same release instead of duplicating it
or leaving a half-created one. The concurrency group is
`release-${{ github.ref }}` with `cancel-in-progress: false` - cancelling a
publish mid-flight is precisely the state to avoid.

## Verified locally

- `uv build` produces `scufris-0.1.0-py3-none-any.whl` and
  `scufris-0.1.0.tar.gz`.
- The smoke test is real: a fresh `uv venv`, `uv pip install` of the built
  wheel with its actual dependencies, then `scufris --version` -> `scufris
  0.1.0`. This is what `--version` was added for in 20260729-125056.
- `./scripts/check-release-ready.sh v0.1.0` passes on a clean tree.
- `./scripts/check-release-ready.sh v9.9.9` fails with
  `version sources disagree: tag says 9.9.9, pyproject.toml and CHANGELOG.md
  say 0.1.0`, exit 1 - the crafted-mismatch proof the Definition of Done asks
  for.
- The guard also caught a genuinely dirty tree during development (its own
  staged changes), which is how the clean-tree check earned its place.

## Checks that exist but do not yet bite

Two of the guard's checks are structurally correct and currently vacuous. They
are worth having - they cost nothing and start working the day the condition
they describe first occurs - but a later reader should not over-trust them:

- **`docs/scratch/` is empty because the directory does not exist.** The drawer
  is a convention introduced by this task (documented in AGENTS.md) and nothing
  writes to it yet. The check becomes meaningful once `/lessons` starts using
  it as its scratch area.
- **The clean-tree check is vacuous on a runner**, where every checkout is
  fresh. It earns its place LOCALLY, stopping an operator from tagging a tree
  with uncommitted changes. The runner-side equivalent is the separate
  assertion that the pinned commit is still the one the tag names.

## What is NOT proven yet

The workflow has never run on a runner - it cannot, until a tag exists or it is
on the default branch where `workflow_dispatch` can reach it. Two DoD items are
therefore outstanding at landing time:

- that pushing a version tag publishes a release with notes and artifacts;
- that re-running for an existing tag does not duplicate it.

Plan: after this lands, dispatch the workflow with a deliberately WRONG version
(`v9.9.9`) as a safe negative proof - the guard must fail on the runner and
publish must never start, with nothing created. The real positive proof belongs
to 20260729-125107, which cuts and pushes `v0.1.0`; its results are to be
recorded back here.
