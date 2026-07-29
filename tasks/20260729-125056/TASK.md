# Make version changelog and release notes a single source of truth

- STATUS: OPEN
- PRIORITY: 95
- TAGS: chore,v0.1.0,release,docs

## Story

As the maintainer, I want the version, the changelog, and the release notes to
come from one place, so that cutting a release is a mechanical step rather than
three files edited by hand and hoped to agree.

`pyproject.toml` has read `version = "0.1.0"` since the first commit, and
`CHANGELOG.md` has an `[Unreleased]` section that has never been closed. Before
a tag can mean anything, those two have to be connected.

## Steps

- [ ] Declare the single source of version truth (`pyproject.toml`) and make
      everything else derive from it: the app's reported version, the package
      metadata, and any UI footer.
- [ ] Expose the running version where it is useful: an API field and the
      dashboard, so the operator can tell what is deployed without reading the
      Nix store path.
- [ ] Define the changelog cut: `[Unreleased]` becomes `[X.Y.Z] - YYYY-MM-DD`,
      a fresh `[Unreleased]` is opened, and link references are updated. Script
      it rather than describing it.
- [ ] Add release-notes extraction: given a version, print exactly that
      section's body for the release pipeline to consume.
- [ ] Add the agreement check: the tag, `pyproject.toml`, and the changelog's
      top released section must name the same version, and the check must fail
      loudly when they do not.
- [ ] Handle the edge cases the scripts will actually meet: a version with no
      changelog section, an empty section, a pre-release suffix, and a re-run
      against an already-cut version.
- [ ] Backfill `CHANGELOG.md` so the accumulated `[Unreleased]` content is
      honestly attributable to v0.1.0 rather than dumped as one entry.

## Definition of Done

- The tag, `pyproject.toml`, and the changelog cannot disagree
  (test: `test_release_version_sources_agree`).
- Release notes for a version can be extracted exactly, including its edge cases
  (test: `test_release_notes_extraction`).
- The running application reports its version
  (test: `test_app_reports_its_version`).
- The changelog cut is scripted and idempotent
  (cmd: `scripts/cut-changelog.sh --check X.Y.Z`).

## Notes

- Epic: 20260729-124706.
- `CHANGELOG.md` already follows Keep a Changelog with Added/Changed/Fixed
  groups. Keep that shape; nova-protocol's subsystem grouping suits a game with
  a news post per release, not this project.
- Scripts live in a `scripts/` directory so CI and a local session run the same
  code, the way nova-protocol's release guards do.

## Flow State

- FLOW STEP: PLANNING
