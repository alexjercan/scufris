# EPIC: Ship tagged releases from CI

- STATUS: OPEN
- PRIORITY: 120
- TAGS: goal,epic,v0.1.0,release,ci

## Epic

Scufris has never been released and has no automation: there is no `.github/`,
no git tag, `pyproject.toml` has sat at `0.1.0` since the first commit, and
`CHANGELOG.md` accumulates an `[Unreleased]` section with nothing that ever
closes it. Every check (`nix flake check`, `cd web && npm run ci`) runs only
when someone remembers to run it locally, across ~480 commits.

There is already a consumer waiting for tags: `~/personal/nix.dotfiles` takes
Scufris as the flake input `github:alexjercan/scufris`, unpinned, so the
operator's own machine tracks whatever `master` happens to be. A release means
that input can pin `github:alexjercan/scufris/v0.1.0` and move deliberately.

Give the project the release shape used in `~/personal/nova-protocol`:
continuous checks on every push and pull request, and a tag-triggered pipeline
that turns `v0.1.0` into a verified, documented GitHub Release. Adapted to what
this project actually is - a Nix-packaged Python app with a webpack frontend,
not a cross-platform game binary - so the artifacts are the Python distribution
and the flake, not per-OS bundles.

## Done Means

1. Every push and pull request runs the full QA gate on a clean checkout:
   `nix flake check` plus the frontend suite (cmd: `gh run list --workflow ci`
   shows green runs on master).
2. Pushing a `vX.Y.Z` tag builds, verifies, and publishes a GitHub Release whose
   notes are that version's `CHANGELOG.md` section, with the built distribution
   attached (manual: the v0.1.0 release page).
3. The git tag, `pyproject.toml`'s version, and the changelog's top released
   section cannot disagree (test: `test_release_version_sources_agree`).
4. A release refuses to publish when the changelog has no section for the tag,
   task records fail `tatr check`, or ephemeral scratch was never compiled into
   `LESSONS.md` (cmd: `scripts/check-release-ready.sh`).
5. The release procedure is written down well enough that a cold session can cut
   the next version unaided (cmd: `rg -n "Releasing" AGENTS.md`).

## Child Tasks

- [x] 20260729-125051 (p100, v0.1.0) add continuous integration for every push
      and pull request
      landed d531d51; 2 review rounds; cold CI run is green in ~2 minutes, so
      no binary cache was needed; proven red by a deliberate ruff+prettier
      break; conformance moved into the flake as a `records` check.
- [x] 20260729-125056 (p95, v0.1.0) make version changelog and release notes a
      single source of truth
      landed fc32e42; 3 review rounds; CHANGELOG.md cut for 0.1.0 here rather
      than in 20260729-125107, so the agreement test asserts a real fact from
      that commit on. Seeded 20260729-140709 ([YANKED] parsing gap).
- [x] 20260729-125101 (p90, v0.1.0) publish a GitHub Release from a version tag
      landed ca231a3; 2 review rounds; guard/verify/publish chain, draft-then-
      publish, VM test kept after a probe found /dev/kvm present-but-unusable.
      Runner-side behaviour still unproven at landing - see its NOTES.md.
- [ ] 20260729-125107 (p25, v0.1.0) document the release procedure and cut
      v0.1.0

## Decisions

- 20260729-125051 DECISION.md: CI runs the real `nix flake check` via
  `DeterminateSystems/nix-installer-action` with no third-party binary cache;
  cold and warm costs are measured and recorded, and adding a cache is a
  separate task if the numbers justify it. (ACCEPTED)
- 20260729-125101 DECISION.md: a release is the tag plus a wheel and sdist from
  `uv build`, attached and smoke-tested before publish; no PyPI; the NixOS VM
  test guards the release only, and is removed rather than skipped if the
  hosted runner has no KVM. (ACCEPTED)

## Landing Scope

The operator approved the full outward-facing run for this epic: land each task
on master, push master, prove CI red-then-green on a scratch branch, and push
the `v0.1.0` tag so a real GitHub Release is produced. The tag push and the
`~/personal/nix.dotfiles` pin are each confirmed with the operator immediately
before they happen; nothing else in the epic pushes without that.

## Manual Acceptance

- (pending) cut v0.1.0: the release page is something the operator would
  actually point another person at.

## Notes

- Reference implementation: `~/personal/nova-protocol/.github/workflows/`
  (`ci.yaml` for the push/PR gate, `release.yaml` for the tag-triggered
  release with a pre-release guard job and `svenstaro/upload-release-action`).
- `CHANGELOG.md` already follows Keep a Changelog with an `[Unreleased]`
  section, so the version-cut step has a defined shape to fill.
- Repository: `git@github.com:alexjercan/scufris.git`, currently no tags.

## Flow State

- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
