# Retro: version, changelog and release notes as one source of truth

- DATE: 20260729
- TASK: 20260729-125056
- REVIEW ROUNDS: 3 (REQUEST_CHANGES, REQUEST_CHANGES, APPROVE)

## What went well

- **Consolidating the two version lookups found a real disagreement.** `app.py`
  and `health.py` each had their own copy with different fallbacks
  (`"0.0.0+unknown"` and `"unknown"`). Nobody had noticed because nothing
  compared them. The task's premise - "make it one source" - was correct for a
  concrete reason, not just tidiness.
- **Cutting the changelog HERE rather than in the tagging task.** Until a
  released section exists, `test_release_version_sources_agree` can only assert
  "nothing cut yet". Cutting 0.1.0 in this task made the epic's named proof
  assert a real fact from this commit onward, and turned the tagging task's step
  into a verification. This is also what forced idempotence to be a real
  property rather than a claim.
- **Fixes landed at the right layer.** The splice follows the parser's offsets;
  newline normalization happens once at the three entry points; the link filter
  compares normalized versions on both sides. No special cases.

## What went wrong

Three rounds, and every finding was the same shape: **code reporting success
while doing the wrong thing.**

- **Round 1: a literal `str.replace` where the parser used a regex.** The two
  disagreed about what a heading is, so `##  [Unreleased]` (two spaces) produced
  a changelog with no released section and no link references - written to disk,
  exit 0, "cut CHANGELOG.md for 1.0.0". The most dangerous bug in the cycle, and
  it was invisible to a green suite.
- **Round 1: the three DoD-named tests were self-referential.** They compared
  the app's reported version against the same call the app makes, so they would
  pass while everything reported `0.0.0+unknown`. The one cross-source assertion
  sat behind an `if` that skipped in exactly that failure mode. The tests named
  in a Definition of Done were the weakest tests in the file.
- **Round 2: my round-1 fix introduced a regression.** Making re-dating
  automatic, combined with `main()` defaulting to today, meant a dateless re-run
  on a later day silently moved a released version's date - breaking the very
  idempotence property the task's DoD names and my own NOTES.md leaned on.
- **Round 2: a test I wrote to pin the fence fix never exercised the fence
  code.** The fixture indented the quoted heading inside a bullet, and the regex
  is anchored `^##`, so it never matched with or without fence awareness. The
  reviewer proved it by disabling fence detection entirely and watching the test
  still pass.

I also wrote one line of genuine nonsense (`return True and False or False if
False else ...` in `is_empty`) and caught it on re-read before running anything.

## Lessons

- `revert-the-fix-to-prove-the-test`: a test written to pin a bug is unproven
  until you revert the fix and watch it fail. Two separate defects this cycle
  survived a green suite - one test never touched the code path it named, and
  one fix regressed a property no test covered. The check costs one edit and one
  run.
- `a-fix-can-break-the-property-it-was-protecting`: after fixing an edge case,
  re-read the invariant the surrounding code claims (here "idempotent", stated
  in the DoD, the script header AND the notes) and test THAT, not just the edge
  case. The re-date fix satisfied the review comment and broke the headline
  property.
- `dod-named-tests-deserve-the-most-scrutiny`: a test named in a Definition of
  Done is the one nobody re-reads, because its name is doing the arguing. Assert
  against an INDEPENDENT source (pyproject.toml), never against the same call
  the code under test makes, and never hide the cross-source assertion behind a
  condition that can silently skip.

## What to do differently next time

Before writing a regex-and-rewrite pair, decide that only ONE of them locates
things. Every round-1 and round-2 defect in the parser traced back to two pieces
of code having different opinions about what a section heading is: a literal
replace vs the regex, then the fence-aware matcher vs `split_document`'s raw
line scan. Locating is a single responsibility; the rewriters should consume its
offsets.

And when a review asks for an escape hatch, add the hatch - do not change the
default. Round 1 asked for a way to re-date; I made re-dating automatic.
