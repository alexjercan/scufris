# Review: Make version changelog and release notes a single source of truth

- DATE: 20260729-140458
- ROUND: 3
- REVIEWER: out-of-context agent
- VERDICT: APPROVE

(Rounds 1 and 2 were REQUEST_CHANGES. See `## Round 3` at the foot for the
current state and the two non-blocking minors left open; the earlier rounds are
kept below as the record.)

## Round 1

Verification performed in the worktree:

- `nix develop --command python -m pytest tests/test_release.py -q` -> 12 passed.
- `ruff check .` -> clean. `mypy .` -> clean (67 files).
- `ruff format --check` -> 2 files would be reformatted (see NIT below).
- Both shell wrappers exercised by hand (`check`, `cut --check` pass and fail
  paths, `notes` pass and missing-version paths, no-args usage) - all exit
  codes and messages correct.
- `scufris --version` run from the dev shell with cwd `/tmp` -> `scufris 0.1.0`,
  rc 0, no config or network needed. `scufris -v --version` also works.
- Parser probes against `scripts.release_tools` for unusual spacing, code
  fences, CRLF, empty documents, missing link blocks, stale link lines, and
  versions containing `+` and `.`.

## Findings

### MAJOR cut_changelog silently no-ops (and eats the link block) when the heading is not the exact literal `## [Unreleased]`

`scripts/release_tools.py:287` mutates the document with
`sections_block.replace(f"## [{UNRELEASED}]", ..., 1)`, a literal string
replace, while `_SECTION_RE` (line 45) deliberately accepts any horizontal
whitespace: `##  [Unreleased]`, `##\t[Unreleased]`. The parser and the mutator
therefore disagree about what a heading is.

Reproduced:

```
doc = "# Changelog\n\nblurb\n\n##  [Unreleased]\n\n- A thing.\n"   # two spaces
out = cut_changelog(doc, "1.0.0", "2026-07-29")
# out == '# Changelog\n\nblurb\n\n##  [Unreleased]\n\n- A thing.\n\n\n'
# -> no [1.0.0] section was created, and the link-reference block is gone
```

Because `out != text`, `main()` (line 356) takes the write branch: it
overwrites `CHANGELOG.md` with a document that has no released section and no
link references, prints `cut CHANGELOG.md for 1.0.0 (2026-07-29)`, and exits 0.
A release pipeline that trusts the exit code has just been told a cut happened
that did not. `_link_lines` compounds it: with `versions == []` it emits nothing
at all, so even the pre-existing `[Unreleased]:` definition is dropped and every
`[x]` reference in the file becomes a dangling link.

Suggested change: do not use `str.replace` at all. `split_document` already has
the match; re-run `_SECTION_RE` over `sections_block`, find the Unreleased
match, and splice on its span. Then assert the result actually changed
(`if cut == sections_block: raise ReleaseError(...)`) so a failed splice can
never be reported as success. A regression test with `##  [Unreleased]` pins it.

### MAJOR the three named DoD tests do not prove the version is right - only that it is consistently whatever it is

The Definition of Done claims "the running application reports its version"
(test: `test_app_reports_its_version`). That test
(`tests/test_release.py:196-211`) asserts
`SCUFRIS_VERSION == scufris_version()`, `app.version == SCUFRIS_VERSION`, and
`health["scufris_version"] == SCUFRIS_VERSION`. Every one of those is the same
call compared against itself. If `importlib.metadata` failed and the app
reported `0.0.0+unknown`, all three assertions still hold and the test is green.
`test_cli_version_flag_prints_the_version` (line 225) has the identical shape:
`assert result.stdout.strip() == f"scufris {scufris_version()}"`.

`test_release_version_sources_agree` (line 52) is the one place the reported
version is compared to `pyproject.toml`, and that comparison is behind
`if __version__ != UNKNOWN_VERSION:` (line 68) - a silent skip, invisible in the
pytest output, in exactly the failure mode it is supposed to catch. The
docstring's claim that the test covers "the installed distribution" is therefore
conditional on the bug not being present.

Suggested change: compare against the independent source. In
`test_app_reports_its_version` and `test_cli_version_flag_prints_the_version`,
assert against `project_version(PYPROJECT.read_text(...))`, not against
`scufris_version()`. If the "never installed" tree is a real supported case for
the suite, express it as an explicit `pytest.skip("scufris is not installed;
version metadata is unavailable")` so a skipped assertion is visible in the run
rather than an `if` that quietly evaporates.

### MINOR a `## [` line inside a fenced code block is parsed as a section heading, silently truncating release notes

`_SECTION_RE` is line-oriented with no fence awareness. Reproduced:

```
"## [1.0.0] - 2026-01-01\n\nExample:\n\n```md\n## [Unreleased]\n```\n\n- real item\n"
parse_changelog -> [Unreleased, 1.0.0(body='Example:\n\n```md'), Unreleased('```\n\n- real item')]
release_notes(doc, "1.0.0") -> 'Example:\n\n```md'
```

The published release page would show a truncated body ending in an unclosed
code fence, and a spurious second `Unreleased` section appears in the section
list. This matters more than the usual "unlikely markdown" case because this
project's changelog documents its own tooling and is a plausible place for a
fenced Keep a Changelog snippet to land.

Suggested change: track fenced regions in `parse_changelog` (a simple
``` ```/~~~ `` toggle over the lines) and ignore heading matches inside them, or
at minimum reject a document containing more than one `[Unreleased]` section as
malformed rather than parsing it into two.

### MINOR CRLF input produces a mixed-line-ending file and `\r` inside the extracted notes

`split_document` (line 216) uses `str.splitlines()` and rejoins with `"\n"`,
while the preamble is sliced out of the original text and keeps its `\r\n`.
Reproduced on a CRLF document: the output is `'...blurb\r\n\r\n## [Unreleased]\n\n## [1.1.0] - ...'`.
Separately, `Section.body` keeps stray `\r` (`body='\r\n- new\r'`), so
`release-notes.sh` emits carriage returns into whatever consumes it.

Not urgent for this repo (the tree is LF), but the file is about to be edited by
a CI job. Suggested change: normalize once at the top of `parse_changelog` /
`cut_changelog` (`text.replace("\r\n", "\n")`), and note in the docstring that
the tool writes LF.

### MINOR an already-cut section can never be re-dated or repaired, and 0.1.0 has been cut to a date no tag exists for

`cut_changelog` returns `text` unchanged as soon as `find_section` sees a cut
section for that version (line 275). That is the right idempotency answer for
the pipeline, but it also means the script has no path to fix a section it
already wrote: a wrong date, or the link block from the MAJOR finding above,
stays wrong forever and the CLI reports `CHANGELOG.md already cut for X`.

This is live right now. `CHANGELOG.md` was cut to `## [0.1.0] - 2026-07-29` and
its links point at `releases/tag/v0.1.0` and `compare/v0.1.0...HEAD`, both 404
until the tag exists - and the tag is child task 20260729-125107's job, which
per the epic includes "cut v0.1.0". If that tag lands on a later date, the
changelog's date is wrong and `scripts/cut-changelog.sh` cannot correct it.

The cut itself is defensible (the commit message argues it makes the agreement
test non-vacuous, which is true, and 0.1.0 has genuinely never been released so
attributing the whole history to it is honest). What is missing is a way out.
Suggested change: add `--force`/`--redate` that re-writes an existing section's
date and rebuilds the links, and either state in 20260729-125107 that the date
must be corrected at tag time or move the date choice there.

### MINOR `_link_lines` preserves stale `v`-prefixed link definitions, emitting duplicates

`_link_lines` (line 241) builds `managed = {UNRELEASED, *versions}` from
NORMALIZED versions, but filters the existing block with the raw
`m.group("version")`. A pre-existing `[v1.0.0]: ...` line is therefore not
recognized as managed and is carried through alongside the freshly generated
`[1.0.0]: ...`. Reproduced - the output contains both. Suggested change: compare
`_normalize(m.group("version"))` against `managed`.

### NIT `ruff format --check` is dirty on both new Python files

`ruff format --diff` rewrites `scripts/release_tools.py` (5 hunks, including a
pointless implicit concatenation at line 283:
`"...release as " f"{wanted}"`) and `tests/test_release.py` (lines 173 and 188
exceed the configured 88-column limit). `nix flake check` only runs
`ruff check`, so this is not gating, but AGENTS.md lists `ruff format .` as part
of the loop. Run it.

### NIT the "Unreleased is not a release" assertion would pass for the wrong error

`tests/test_release.py:113` uses `pytest.raises(ReleaseError, match="Unreleased")`.
The "no section for X" message at `release_tools.py:142` also contains the word
`Unreleased` when `X == "Unreleased"`, so this assertion cannot distinguish the
behavior it is testing from a lookup failure. Match on
`"resolves to the \\[Unreleased\\] section"` instead.

### NIT no NOTES.md for the shipped change

AGENTS.md ("Where records go") makes `tasks/<id>/NOTES.md` the design/fix record
for a shipped change, and the global AGENTS.md asks for what changed, what was
hard, and self-reflection. `tasks/20260729-125056/` contains only `TASK.md`. The
commit message carries some of it; the parser edge cases found above are exactly
the kind of thing a future session would want written down.

### NIT `scufris serve --version` is rejected, and `scripts` is a very generic top-level package name

`--version` is on the root parser only, so `scufris serve --version` exits 2
with "unrecognized arguments". Harmless for the smoke test, mildly surprising
given `-v/--debug` is threaded through every subparser via `parents=[common]`.
Separately, `scripts/__init__.py` claims the top-level import name `scripts`;
the wheel is safe (`[tool.hatch.build.targets.wheel] only-include = ["scufris"]`,
verified), but it is a name with a high collision surface on `sys.path`.

## What is good

- The version consolidation is clean and complete. `scufris/version.py` is a
  genuinely single source; both old copies are gone; the surviving fallback is
  the PEP 440 local version, and grepping the tree confirms nothing depended on
  the discarded `"unknown"` string (`web/src/settings-view.ts` and
  `scufris/telegram.py` just render whatever the health field says).
- `scufris --version` verified to work from a subprocess with no config, no
  network, no backend - argparse exits inside `parse_args`, before `Settings()`
  - and it does not disturb `_wants_debug`, which scans argv independently.
- Version strings containing regex metacharacters are safe by construction: no
  regex is ever built from a version. Confirmed with `1.0.0+a.b`.
- The empty-changelog, no-link-block, empty-section, undated-section, and
  pre-release-suffix paths all behave correctly and fail with messages that name
  the fix. `--check` deliberately bypassing `cut_changelog` (with the comment
  explaining why) is the right call and reports the true answer.
- Error messages name every value seen rather than saying "mismatch", the
  wrappers exit 2 on usage errors and 1 on invariant violations, and
  `set -euo pipefail` plus the empty-array guard are correct.
- The comment at `_SECTION_RE` explaining why `[^\S\n]` rather than `\s` records
  a real bug that was hit and fixed - exactly the kind of note that stops a
  future session from "simplifying" it back.
- ASCII punctuation throughout, no AI attribution in the commit.

## Round 2

- DATE: 20260729-135512
- COMMIT REVIEWED: b20c53a (net diff `master...HEAD`)
- VERDICT: REQUEST_CHANGES

### Verification performed

- `ruff check .` clean; `mypy .` clean (67 files); `python -m pytest
  tests/test_release.py -q` -> 17 passed. `ruff format --check .` names only
  `scufris/enums.py`, `tests/test_mcp_server.py`, `tests/test_supervisor.py` -
  all pre-existing on master, none touched by this branch. Non-ASCII sweep of
  the new and changed files: clean.
- Each round-1 fix was reverted in a scratch copy of the tree and the suite
  re-run, to check the new tests actually pin the behavior. Four of the five do:
  `test_cut_handles_an_oddly_spaced_unreleased_heading`,
  `test_crlf_input_produces_clean_lf_output`,
  `test_an_already_cut_section_can_be_redated` and
  `test_cut_replaces_a_stale_v_prefixed_link_reference` all fail when their fix
  is reverted. The fifth does not - see MAJOR below.
- New parser probes: indented fences, a 4-tilde fence with an info string,
  nested `~~~` inside ``` ``` ```, an unterminated fence in the preamble and in
  `[Unreleased]`, a fenced link block at the foot of the file, `_redate` against
  an oddly-spaced heading and against a `v`-prefixed heading, and `main()`'s
  date defaulting with a faked `date.today()`.

### Round-1 findings: status

| Round-1 finding | Status |
|---|---|
| MAJOR literal-replace cut | Fixed. The splice now uses `heading.end()` from `_section_matches`, and a first section that is not `[Unreleased]` is refused outright. Pinned by a test that fails when reverted. |
| MAJOR self-referential DoD tests | Fixed. All three now compare against `project_version(pyproject.toml)`, and the `if __version__ != UNKNOWN_VERSION` skip is an unconditional assert with a message that says how to run the suite properly. |
| MINOR code fence | Behavior fixed and verified by hand; the test that claims to prove it does not. See MAJOR below. |
| MINOR CRLF | Fixed. `_normalize_newlines` also folds bare `\r`. Pinned. |
| MINOR cannot re-date | Fixed at the library level, but the CLI default it introduced is a new problem. See MAJOR below. |
| MINOR stale `v`-prefixed link | Fixed. Pinned. |
| NIT ruff format | Fixed on the new files. |
| NIT loose `pytest.raises` pattern | Fixed. |
| NIT no NOTES.md | Fixed, and it is a good one - it records the bugs and where they came from. |
| NIT `scufris serve --version` | Deliberately declined and recorded as a known limitation. Accepted. |

### MAJOR `cut-changelog.sh <version>` with no `--date` silently re-dates an already-released section

`main()` at `scripts/release_tools.py:422` computes
`date = args.date or _datetime.date.today().isoformat()` and hands it to
`cut_changelog`, which since round 2 re-dates any existing section whose date
differs (line 331-334). So the date argument is never actually optional: on any
day after the cut, a dateless re-run rewrites the released section's date.

Reproduced with a faked `date.today()`:

```
main([... "cut", "1.0.0", "--date", "2026-07-29"])  -> cut CHANGELOG.md for 1.0.0 (2026-07-29)
# same command, no --date, on 2026-08-15:
main([... "cut", "1.0.0"])                          -> cut CHANGELOG.md for 1.0.0 (2026-08-15)
# file now says: ## [1.0.0] - 2026-08-15
```

`check` still passes afterwards, so nothing downstream notices that a published
version's release date moved. This contradicts the Definition of Done ("The
changelog cut is scripted and idempotent") and `scripts/cut-changelog.sh`'s own
header, which still advertises `scripts/cut-changelog.sh 0.1.0` as "perform the
cut (idempotent)". It also contradicts NOTES.md, which argues the cut had to be
idempotent precisely so the tagging task can re-run it.

Round 1 asked for an explicit escape hatch, not for re-dating to become the
default of the bare command. Suggested change: only re-date when the operator
said so. Pass `args.date` (possibly `None`) into `cut_changelog`; use today's
date for a FRESH cut, and for an already-cut section re-date only when a date
was supplied explicitly and differs, otherwise no-op and print "already cut".
Add a test over `main()` (not just `cut_changelog`) that runs the dateless cut
twice with different `today` values and asserts the file is byte-identical -
that is the invariant the DoD names, and nothing currently covers `main()`'s
date defaulting.

### MAJOR `test_a_heading_inside_a_code_fence_is_not_a_section` does not exercise the fence code at all

`tests/test_release.py:218-234`. The fenced heading in the fixture is indented
by two spaces (`  ## [9.9.9] - 2099-01-01`, inside a bullet), and `_SECTION_RE`
is anchored `^##` with no leading-whitespace allowance - so that line never
matched, fence awareness or not. Confirmed by setting `spans = []` in
`_section_matches` (fence detection fully disabled) and re-running: this test
still passes, while the other four new tests fail. Confirmed the other way too:

```
fence-blind parse, indented fixture   -> ['Unreleased', '1.0.0']   # test passes
fence-blind parse, unindented fixture -> ['Unreleased', '1.0.0', '9.9.9']
```

The `_fenced_spans` implementation itself is correct - I verified independently
that an unindented ` ```md ` block, a 4-tilde fence with an info string, and a
`~~~` marker nested inside a ``` ``` ``` block are all handled right. The
problem is only that the named proof proves nothing, which is the same class of
finding as round 1's MAJOR on the DoD tests, recurring on the new code.

Suggested change: put the fence at column 0 in the fixture (a top-level
` ```markdown ` block between two bullets is realistic for this changelog), so
`^##` genuinely matches inside it. While there, add the case below.

### MINOR an unterminated fence hides every later section, and the cut then writes a wrong file with exit 0

`_fenced_spans` (line 127) treats an unterminated fence as running to the end of
the document. That is a defensible reading, but it makes an entire changelog
tail invisible and `cut_changelog` then acts on the truncated picture without
complaint. Reproduced with an unclosed ` ```py ` in `[Unreleased]` and a real
`## [1.0.0]` section below it:

- `parse_changelog` returns only `['Unreleased']`.
- `cut 1.1.0` emits `[1.1.0]: .../releases/tag/v1.1.0` - the "oldest version"
  form - even though 1.0.0 exists and it should be a compare link.
- The stale `[1.0.0]: http://x/t` line is preserved verbatim instead of being
  regenerated, because 1.0.0 is no longer in `managed`.
- The real `## [1.0.0]` heading and body are now inside 1.1.0's section, so
  `release-notes.sh 1.1.0` would publish the previous release's notes too.
- Exit code 0, message `cut CHANGELOG.md for 1.1.0`.

This is the same fail-silent shape as round 1's blocker, moved to a new trigger.
Suggested change: make an unterminated fence an error -
`raise ReleaseError("CHANGELOG.md has an unterminated code fence at line N")` -
in `_fenced_spans` or in a small `validate` helper the cut calls. A malformed
changelog should stop the release, not quietly reshape it. Add it to the fence
test above.

### NIT dead fallback in `_redate`, and a fence-blind tail scan in `split_document`

`_redate` (line 378) ends with a bare `return text` when no heading matched. Its
only caller has already located the section via `parse_changelog`, so the branch
cannot legitimately fire; if it ever did, `main()` would print "already cut" for
a re-date that silently did nothing. Prefer raising, or drop the loop in favour
of indexing the match the caller already has.

Separately, `split_document`'s trailing-link scan (line 268) walks raw lines with
`_LINK_RE.fullmatch` and is not fence-aware, unlike `_section_matches`. I could
not turn this into a real failure - a fenced example ending in a link line is
followed by the closing fence marker, which breaks the scan correctly (verified)
- so this is only a consistency note: the two halves of the parser now use
different rules for "is this line real".

One more probe for the record: `_fenced_spans` uses `line.lstrip()[:3]`, so a
fence marker that shares a line with list punctuation (`- ```py`) is not
recognized as a fence. That is the correct CommonMark reading (a fence must be
the first content on its line), and an indented fence inside a bullet IS
recognized because `lstrip` removes the indent. Both verified; no action needed.

### What is good in round 2

- Every round-1 fix was made at the right layer rather than special-cased: the
  splice follows the parser's offsets, newline normalization happens once at the
  three entry points, and the link filter compares normalized versions on both
  sides.
- The refusal `"CHANGELOG.md's first section is not [Unreleased]; refusing to
  cut"` is exactly the right instinct - the round-1 bug was a silent success,
  and the replacement fails loudly.
- `_fenced_spans` is more careful than it needed to be: it tracks WHICH marker
  opened the fence, so a `~~~` inside a ``` ``` ``` block does not close it.
  Verified.
- The comments explaining why each fix is shaped the way it is (the splice
  comment at line 345, the `_link_lines` comment at line 291, the test comment
  about the loose `Unreleased` pattern) record the failure mode, not just the
  fix. NOTES.md's "Bugs found and fixed along the way" is the same discipline and
  is genuinely useful for a cold session.
- The unconditional `assert __version__ != UNKNOWN_VERSION` with an actionable
  message is a better answer than the skip it replaced.

## Round 3

- DATE: 20260729-140458
- COMMIT REVIEWED: bb92560 (merge base with master is 271fea1)
- VERDICT: APPROVE

### Verification performed

- `ruff check .` clean; `mypy .` clean (67 files); `python -m pytest
  tests/test_release.py -q` -> 19 passed. `ruff format --check .` still names
  only the 3 files that were already unformatted on master, none of them touched
  by this branch. Non-ASCII sweep of every file this branch adds or changes:
  0 hits.
- The DoD commands re-run against the live tree: `scripts/cut-changelog.sh
  --check 0.1.0` -> "CHANGELOG.md is cut for 0.1.0"; `scripts/release-notes.sh
  v0.1.0` -> the 0.1.0 body; `release_tools check v0.1.0` -> "version sources
  agree on 0.1.0".
- Each round-3 fix was reverted in a scratch copy and the suite re-run. All
  three are genuinely pinned, which is the check round 2 showed was necessary:
  - restoring the round-2 `if existing.date == date` (re-date by default) fails
    `test_an_already_cut_section_can_be_redated_only_on_request` AND
    `test_a_dateless_rerun_of_the_cut_command_never_moves_the_date`;
  - restoring `spans.append((open_at, len(text)))` for an unterminated fence
    fails `test_an_unterminated_code_fence_is_an_error_not_a_silent_truncation`;
  - setting `spans = []` (fence detection off) now fails BOTH fence tests, where
    in round 2 it failed neither. The author's claim checks out.
- New probes: a fence opened in the preamble and closed after the first heading,
  a `[YANKED]` suffix, a nonsense `--date`, a version heading that appears
  twice, and `--check` / `notes` against a file with an unterminated fence.

### Round-2 findings: status

| Round-2 finding | Status |
|---|---|
| MAJOR dateless re-run re-dates | Fixed correctly and at the right seam. `redate` is opt-in, `main()` sets it only from an explicit `--date`, and `today()` is a named module-level function so the test can pin two different days and assert the file is byte-identical. The wrapper's header comment was corrected too, which I had flagged. |
| MAJOR fence test proved nothing | Fixed. The fixture is at column 0 and the docstring records WHY the indentation mattered, so the next person does not re-indent it. Verified by disabling fence detection. |
| MINOR unterminated fence | Fixed. `_fenced_spans` raises and names the line the fence opened on. `--check` and `notes` against such a file now exit 1 with that message rather than a misleading one. |
| NIT `_redate` dead fallback | Fixed - raises "vanished mid-edit" instead of reporting a successful no-op. |
| NIT fence-blind `split_document` | Declined by the author. **I agree.** I tried again: a fenced example whose lines look like link references is always followed by its closing fence marker, which breaks the tail scan correctly; and the unbalanced case now raises before the scan is reached. There is no reachable failure, and adding fence tracking to the tail scan would be complexity with no behavior behind it. Correct call. |

I also re-checked the round-1 fixes still hold after this round's edits: they do.

Two things worth recording as good judgement rather than just compliance. First,
`_fenced_spans` raising means `parse_changelog` can now raise where it used to
return, and every caller reaches it inside `main()`'s `try` - verified that
`--check` and `notes` on a malformed file exit 1 naming the fence, not a
misleading "not cut for X". Second, a fence opened in the preamble and closed
after the first heading (the one substring-parity case I was worried about)
fails loudly and correctly with "no [Unreleased] section to cut", because the
heading really is inside the fence - the CommonMark reading and the tool's
reading agree.

### MINOR a `[YANKED]` suffix makes the whole section disappear

`_SECTION_RE` (`scripts/release_tools.py:45`) ends `(?P<date>\S+))?[^\S\n]*$`,
so anything after the date breaks the match outright. Keep a Changelog 1.1.0 -
the spec this file's own header links to - documents exactly
`## [0.0.5] - 2014-12-13 [YANKED]`. Reproduced:

```
## [Unreleased] / ## [1.0.0] - 2026-01-01 [YANKED] / ## [0.9.0] - 2025-01-01
parse_changelog -> [('Unreleased', None), ('0.9.0', '2025-01-01')]
```

The yanked section does not merely go missing - its heading and body are
absorbed into `[Unreleased]`'s body, so `[Unreleased]` silently looks non-empty
and a subsequent cut would splice above it and regenerate link references that
skip the yanked version. `release_notes` for it says "has no section for 1.0.0.
Cut it first", which sends the operator to a command that will not help.

Not blocking: `check_agreement` fails loudly on the release-critical path (it
would report the top released section as 0.9.0 against a 1.0.0 pyproject), and
nothing in this repo is yanked. Suggested change when convenient: allow an
optional trailing bracketed marker in `_SECTION_RE` and carry it on `Section`
(`yanked: bool`), so a yanked release parses and `release_notes` can refuse it
for the right reason.

### MINOR `--date` is not validated, and it is now the flag that authorizes re-dating

`main()` passes `args.date` straight through. Reproduced:

```
release_tools cut 1.0.0 --date bananas   -> cut CHANGELOG.md for 1.0.0 (bananas)
# CHANGELOG.md now reads: ## [1.0.0] - bananas
release_tools check v1.0.0               -> version sources agree on 1.0.0   (rc 0)
release_tools notes 1.0.0                -> the notes                        (rc 0)
```

Nothing downstream objects, because `_SECTION_RE`'s date group is `\S+` and
`check_agreement` only tests `date is not None`. This mattered less before
round 3; now `--date` is the single gesture that authorizes overwriting an
already-published version's date, so a fat-fingered value both re-dates and
installs nonsense in one step.

Not blocking - the default path never produces a bad date, and it takes an
operator typo to reach. Suggested change: `datetime.date.fromisoformat(args.date)`
in `main()`, raising `ReleaseError(f"--date must be YYYY-MM-DD, got {args.date}")`,
and optionally the same assertion on the top section's date inside
`check_agreement` so a hand-edited changelog cannot smuggle one in either.

### Why this is APPROVE

Every claim in the Definition of Done is now backed by a test that fails when
its fix is removed - I checked that mechanically for all eight fixes across the
three rounds rather than taking the green suite as proof. The two findings above
are genuine but neither is reachable on the path this epic actually walks: the
default `cut` produces an ISO date, the live changelog has no fences and nothing
yanked, and the release-critical `check` fails loudly in both scenarios. They
belong in a follow-up, not in another round here.

### What is good in round 3

- The fix for the idempotence regression is at the seam that caused it. Round 2
  fixed `cut_changelog` and left `main()`'s date defaulting untested; round 3
  introduced `today()` specifically so the test could drive `main()`, which is
  where the DoD's invariant actually lives. Testing the layer that broke, not
  the layer that was convenient, is the right instinct.
- The unterminated-fence error names the line number. That is the difference
  between a usable message and one that sends someone hunting through a
  200-line changelog.
- Docstrings and comments now carry the counterfactual, not just the rule: the
  fence test explains why the fixture must stay unindented, `cut_changelog`
  explains why re-dating is off by default, and `main()` explains why `--date`
  is what authorizes it. That is what stops the next session from "simplifying"
  a fix back into the bug.
- NOTES.md's findings 7 to 9 draw the right lesson out loud - that a fix can
  introduce a regression and that a test written to pin a bug should be checked
  by reverting the fix. That generalizes beyond this task and is worth carrying
  into LESSONS.md.
- The author disagreed with one review point (`split_document`) with a stated
  reason instead of complying by reflex, and the reason was correct.
