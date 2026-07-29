# Review: EPIC - Ship tagged releases from CI

- DATE: 20260729-150000
- ROUND: 1
- REVIEWER: acceptance pass against the epic's Done Means
- VERDICT: APPROVE

This is not a code review - each child task carried its own out-of-context
review (7 rounds in total across the four, 5 of them REQUEST_CHANGES). This
record is the acceptance check on the CONTAINER: does the delivered thing meet
what the epic said it would, and what did not get done.

## Done Means, checked

### 1. Every push and pull request runs the full QA gate on a clean checkout

MET. `.github/workflows/ci.yaml`, two jobs. Green master runs: 30445778357
(d531d51), 30446474687 (fc32e42), plus the runs for 5da30c3 and 339ea27.
Verified as `event: push`, not only on pull requests. Proven to DISCRIMINATE,
not merely to pass: run 30443929343 carried a deliberate ruff + prettier break
and both jobs went red; the revert went green again.

Delivered beyond the plan: repository conformance (`tatr check --ledger
LESSONS.md`) moved INTO the flake as a `records` check, so the same gate runs
locally and on the runner instead of existing only in CI.

### 2. Pushing a vX.Y.Z tag builds, verifies and publishes a GitHub Release

MET. Run 30449339746 on tag v0.1.0, green in ~7.5 minutes across guard, full
gate on the tagged commit (including the NixOS VM test), and
build-smoke-publish. Page: https://github.com/alexjercan/scufris/releases/tag/v0.1.0
with notes from the changelog section and both artifacts attached.

Verified from OUTSIDE the pipeline: the wheel downloaded from the release page
installs into a fresh virtualenv and reports `scufris 0.1.0`.

### 3. The git tag, pyproject.toml's version and the changelog cannot disagree

MET. `test_release_version_sources_agree` asserts it against the live tree and
runs inside `nix flake check`, so it gates locally, in CI, and in the release
pipeline. It compares against an INDEPENDENT source (`pyproject.toml`) after
round 1 of that task found all three DoD-named tests were comparing the app to
itself.

### 4. A release refuses to publish when the changelog, task records or scratch are bad

MET. `scripts/check-release-ready.sh` - five checks, each printing what it
verified. Proven red locally on a crafted version mismatch (`v9.9.9`, exit 1)
and on a dirty tree (twice, unintentionally, during development). Proven to
block the pipeline ON THE RUNNER: run 30448350452 failed at the guard and both
downstream jobs SKIPPED, with no release and no tag created afterwards.

Honest limit: that runner-side run stopped at tag checkout, so the guard
SCRIPT's version-disagreement path has only ever been exercised locally. The
real v0.1.0 run exercised the script itself, passing.

### 5. The release procedure is written down well enough for a cold session

MET, and tested by use: `rg -n "Releasing" AGENTS.md`, and the v0.1.0 release
was cut by following it literally. Round 1 of that task found a BLOCKER in the
procedure - it tagged before pushing master and never said which checkout to
run from - which is exactly the failure a cold session would have hit.

## What was NOT delivered

- **The operator's machine is not running the release.** `~/personal/nix.dotfiles`
  is pinned and committed locally (0b20eb8), but not pushed and not rebuilt, per
  the operator's instruction. The epic's second manual acceptance item stays
  open. What is proven is that the artifact and the flake pin work, not that the
  running dashboard reports 0.1.0.
- **Idempotence was proven for the update path, not the recovery path.**
  Re-running for v0.1.0 (run 30449912896, via workflow_dispatch) left exactly
  one release with the same assets and the same publishedAt. The
  failed-publish-leaves-a-draft path is reasoned and reviewed but has never
  actually failed, because nothing failed.
- **`docs/scratch/` is enforced but empty.** The drawer is a convention this
  epic introduced; the guard check for it is structurally correct and currently
  vacuous. Recorded as such in tasks/20260729-125101/NOTES.md.

## Follow-ups seeded

- 20260729-140709 - the changelog parser drops a `[YANKED]` section entirely
  (found by review, off the v0.1.0 path, filed rather than fixed).
