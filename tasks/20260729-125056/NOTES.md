# Notes: version, changelog and release notes as one source of truth

- DATE: 20260729
- TASK: 20260729-125056

## What shipped

- `scufris/version.py` - the single runtime lookup of the version.
  `pyproject.toml` is the source; the installed distribution's metadata is that
  same string seen from the other side of packaging, so nothing parses TOML at
  runtime. `app.py` and `health.py` each had their own copy of this lookup with
  DIFFERENT fallbacks (`"0.0.0+unknown"` and `"unknown"`); both now call the one
  helper, and `UNKNOWN_VERSION` is the single sentinel.
- `scufris --version`, which the release pipeline smoke-tests against a freshly
  built wheel. Long form only: `-v` is already `--debug`, and the argparse
  `version` action prints and exits before any config, network or backend is
  touched.
- `scripts/release_tools.py` - the changelog plumbing: parse Keep a Changelog,
  extract one version's notes exactly, cut `[Unreleased]` into a dated section,
  and assert the tag / `pyproject.toml` / changelog agree. `cut-changelog.sh`
  and `release-notes.sh` are thin wrappers so the release procedure has stable
  command names.
- `CHANGELOG.md` cut for 0.1.0, with generated link references.

## Why the cut happened here and not in the tagging task

`test_release_version_sources_agree` is the epic's named proof. Until a
released section exists it can only assert "no released section yet", which
proves nothing about this tree. Cutting 0.1.0 now makes the test assert a real
fact from this commit onward, and the tagging task re-runs the same script as a
verification (`--check`) instead of a first cut. That is exactly why the cut had
to be idempotent.

The 0.1.0 section is dated but NOT tagged yet, so later tasks in this epic still
add their entries to it. That is legitimate: the section is only frozen once the
tag exists. `cut_changelog` can re-date an already-cut section, so if the tag
slips a day the date can be corrected rather than being stuck at its draft
value.

## Bugs found and fixed along the way

1. **The section regex ate the next line.** `^##\s+\[(...)\]\s*(?:-\s*(\S+))?$`
   with `re.MULTILINE` - `\s` matches newlines, so the optional date group
   reached across the blank line and captured the section's first list item as
   the date. An undated section looked dated and an empty one looked full.
   Found by two failing tests on the first run. Fixed by using `[^\S\n]`
   (horizontal whitespace) throughout.
2. **The cut spliced on a literal string.** The parser accepted any horizontal
   whitespace in `##  [Unreleased]`, but the rewrite used
   `str.replace("## [Unreleased]", ...)`. On an oddly-spaced heading the replace
   matched nothing, and the function still returned (and `main()` still wrote)
   a changelog with NO released section and NO link references, reporting
   success. Found by round-1 review. The splice now uses the offsets the parser
   found, and refuses outright if the first section is not `[Unreleased]`.
3. **The DoD tests were self-referential.** `test_app_reports_its_version` and
   the CLI test compared the reported version against `scufris_version()` - the
   same call the app makes - so they would pass while everything reported
   `0.0.0+unknown`. The one cross-source assertion was behind
   `if __version__ != UNKNOWN_VERSION:`, an invisible skip in precisely the
   failure mode it guarded. Found by round-1 review. All three now compare
   against the version parsed out of `pyproject.toml`, and the missing-metadata
   case fails loudly with an instruction instead of skipping.
4. **A `## [` inside a code fence parsed as a section**, truncating the real
   notes at the fence and inventing a phantom section. Fixed with fence-aware
   matching.
5. **CRLF input** produced mixed line endings and `\r` inside extracted notes.
   Normalized on parse.
6. **Stale `[v1.0.0]:` link references** survived next to generated `[1.0.0]:`
   ones because the preserve-filter compared raw strings against normalized
   ones. Now compared normalized.

7. **The re-dating fix broke idempotence** (round 2). Making `cut_changelog`
   re-date any section whose date differed, combined with `main()`'s
   `date = args.date or today()`, meant a DATELESS re-run on a later day
   silently moved a released version's date and exited 0 - the opposite of the
   DoD's "scripted and idempotent", and a regression introduced by a fix.
   Re-dating is now opt-in (`redate=True`, set only when `--date` is passed),
   and the invariant is pinned by a test over `main()` with a monkeypatched
   `today()`.
8. **A test that proved nothing** (round 2). The code-fence test's fixture put
   the quoted heading inside a bullet, indented two spaces - and `_SECTION_RE`
   is anchored `^##`, so that line never matched whether fences were understood
   or not. The reviewer proved it by disabling fence detection entirely and
   watching the test still pass. The fixture is now at column 0, and disabling
   `_fenced_spans` fails both fence tests.
9. **An unterminated fence silently truncated the document** (round 2). Every
   section below it became invisible, so a cut regenerated the wrong links and
   folded the previous release's notes into the new section - reporting success.
   Now a `ReleaseError` naming the line the fence opened on.

Findings 2 through 6 all came from the out-of-context round-1 review, which is
the strongest argument for that step in this cycle: every one of them was a
case where the code reported success while doing the wrong thing, and the local
gate was green throughout. Findings 7 to 9 came from round 2 and are worse in
kind: 7 is a regression introduced BY a round-1 fix, and 8 is a test written to
prove a round-1 fix that did not actually touch the code path. Both were found
by reverting the fix and re-running the suite - a check worth doing on any test
written to pin a bug, not just when a reviewer asks.

## Known limitation, deliberately not fixed

`scufris serve --version` is rejected - the flag is on the top-level parser
only, not on each subparser. The release smoke test uses `scufris --version`,
and putting a version flag on every subcommand is noise for no gain. Recorded
rather than silently ignored.
