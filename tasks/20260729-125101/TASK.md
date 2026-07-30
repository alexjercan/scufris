# Publish a GitHub Release from a version tag

- STATUS: CLOSED
- PRIORITY: 90
- TAGS: infra,v0.1.0,release,ci
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the maintainer, I want pushing a `vX.Y.Z` tag to produce a published GitHub
Release, so that shipping is one deliberate act and the operator's NixOS
configuration can pin a version instead of tracking master.

## Steps

- [x] Add `.github/workflows/release.yaml` triggered on tags matching
      `v[0-9]+.[0-9]+.[0-9]+*`, plus a manual dispatch with an explicit version
      input, following the nova-protocol shape.
- [x] Add the pre-release guard job that everything else depends on, so a
      failing guard blocks the whole release: version agreement, a changelog
      section for this version, `tatr check --ledger LESSONS.md` clean, and no
      uncompiled ephemeral scratch.
- [x] Run the full gate on the tagged commit rather than trusting that CI passed
      on master. Attempt the NixOS VM test here (and only here); if the hosted
      runner has no `/dev/kvm`, REMOVE the step and record that finding in this
      task rather than leaving a step that skips itself and reports success
      (DECISION.md).
- [x] Build the artifacts: the Python distribution (`uv build`: wheel and sdist)
      and a verified `nix build .#scufris`, so a release proves the flake it
      claims to ship. Nothing is published to PyPI (DECISION.md).
- [x] Publish the GitHub Release: title the tag, body the extracted changelog
      section, artifacts attached, and pre-release suffixes marked as
      pre-releases.
- [x] Make it idempotent and safe to re-run: re-running against an existing tag
      updates rather than duplicating, and a failed publish leaves no
      half-created release. (Round-1 review: the first draft published BEFORE
      uploading assets, so an upload failure left a live, empty, watcher-notified
      release. Now created as a draft, filled, then flipped visible last.)
- [x] Verify the installed artifact actually runs (`scufris --version` from the
      built distribution) before publishing, not after.

## Definition of Done

- Pushing a version tag publishes a release with notes from the changelog and
  the artifacts attached (manual: the first real release page).
- A tag whose version disagrees with `pyproject.toml` or has no changelog
  section fails the guard and publishes nothing
  (cmd: `scripts/check-release-ready.sh vX.Y.Z` fails on a crafted mismatch).
- The published artifact runs (cmd: the workflow's smoke step).
- Re-running the workflow for an existing tag does not duplicate the release
  (manual: verified once, recorded in the task).

## Notes

- Epic: 20260729-124706.
- Depends on: the CI workflow and the version/changelog task.
- Reference: `~/personal/nova-protocol/.github/workflows/release.yaml` - the
  `guard-docs` job gating `get-version`, and `svenstaro/upload-release-action`
  for asset upload.
- The real consumer is `~/personal/nix.dotfiles`, which currently takes
  `github:alexjercan/scufris` unpinned. A release exists so that input can name
  a tag.
