# Changelog parser drops a [YANKED] section entirely

- PRIORITY: 0
- TAGS: backlog, release, bug
- ACTIVITY: UNDERSTANDING
- GATES: -
- RESOLUTION: -

## Story

As the maintainer, I want a yanked release to stay visible to the changelog
tooling, so that pulling a bad version does not silently corrupt the next cut.

Keep a Changelog 1.1.0 - which `CHANGELOG.md`'s own header links to - documents
`## [1.0.0] - 2026-01-01 [YANKED]` as the way to mark a pulled release.
`_SECTION_RE` in `scripts/release_tools.py` does not match that form at all, so
the heading and its whole body are absorbed into the PREVIOUS section's body.

Found by the round-2/3 out-of-context review of task 20260729-125056, with a
reproduction. Not on the path the v0.1.0 release walks (nothing is yanked, and
`check_agreement` fails loudly on the release-critical path), so it was filed
rather than fixed in that task.

## Steps

- [ ] Reproduce first: a test with a `[YANKED]` section that shows it vanishing
      into the section above it.
- [ ] Teach `_SECTION_RE` the optional trailing `[YANKED]` marker, and carry it
      on `Section` so callers can tell a yanked release from a live one.
- [ ] Decide and record what `release_notes`, `check_agreement` and
      `_link_lines` should do with a yanked version - in particular whether the
      top released section being yanked should block a release.
- [ ] Check the neighbouring assumptions while in there: `--date` validation
      landed with 20260729-125056, but confirm nothing else parses a heading by
      hand.

## Definition of Done

- A `[YANKED]` section parses as its own section and does not pollute its
  neighbour (test: `test_a_yanked_section_is_still_a_section`).
- Cutting a new version above a yanked one generates correct link references
  (test: covered by the same test module).
- The decision about whether a yanked top section blocks a release is written
  down, not implied (manual: the user reads the recorded decision).

## Notes

- Reported in `tasks/20260729-125056/REVIEW.md` (round 3, non-blocking MINOR).
- The other round-3 MINOR (`--date` unvalidated) WAS fixed in 20260729-125056.
