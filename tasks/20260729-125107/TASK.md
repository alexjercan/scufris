# Document the release procedure and cut v0.1.0

- STATUS: OPEN
- PRIORITY: 25
- TAGS: docs,v0.1.0,release

## Story

As the operator, I want the release procedure written down and v0.1.0 actually
cut, so that the mechanism is proven by use and my machine runs a version I
chose rather than whatever master was that morning.

This is the last task of the release: everything else in v0.1.0 is done before
it, and cutting the tag is what closes the milestone.

## Steps

- [ ] Write the `Releasing` section in AGENTS.md: when to bump, how to cut the
      changelog, how to tag, what the guards check, what to do when the pipeline
      fails halfway, and how to yank a bad release.
- [ ] Update README with the CI badge, the released-version story, and how to
      consume Scufris as a pinned flake input.
- [ ] Run the release readiness checks locally and fix whatever they surface:
      task records clean under `tatr check`, scratch compiled into `LESSONS.md`,
      changelog honest about what v0.1.0 contains.
- [ ] Cut the changelog for 0.1.0, confirm `pyproject.toml` agrees, and tag
      `v0.1.0`.
- [ ] Watch the pipeline through to a published release and verify the page:
      notes, artifacts, and a runnable distribution.
- [ ] Pin the release in `~/personal/nix.dotfiles`: change the `scufris` flake
      input to the tag, rebuild, and confirm the running service reports the
      released version.
- [ ] Record what the first release actually cost and what broke, so the second
      one is boring.

## Definition of Done

- The release procedure is documented end to end (cmd: `rg -n "Releasing" AGENTS.md`).
- `v0.1.0` exists as a tag and as a published GitHub Release with notes and
  artifacts (cmd: `gh release view v0.1.0`).
- The operator's machine runs the pinned release and reports its version
  (manual: the dashboard shows `0.1.0` after a rebuild against the pinned input).
- manual: the release page is something the operator would point another person
  at.

## Notes

- Epic: 20260729-124706.
- Depends on: every other v0.1.0 task. This is the closing act, not a parallel
  one.
- Pinning the dotfiles input is a change to a SECOND repository; it is the first
  real exercise of the host operator epic's configuration flow if that has
  landed by then.

## Flow State

- FLOW STEP: PLANNING
