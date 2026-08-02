# Adopt flow v2: root LESSONS.md, clean tatr check, AGENTS.md flow section

- PRIORITY: 90
- TAGS: chore, process
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a repo in the flow ecosystem, I want the v2 /flow conventions in place -
root LESSONS.md ledger, clean tatr check, AGENTS.md pointing at /flow - so
development here compounds the same way as everywhere else. Part of the
six-repo adoption goal (umbrella: nix.dotfiles tasks/20260720-171807).

## Steps

- [x] Ledger at the root: move docs/LESSONS.md to LESSONS.md (git mv) - or
      create it from the lessons-skill format if the repo has none - then
      run the doc-surface sweep for every reference to the old path
      (AGENTS.md, README, scripts, CI guards, wiki pages) and update them.
      Bring the ledger to format: bare counts until promotion, a
      "## Pending promotions (3+ occurrences, user decides)" section;
      move unpromoted (x3)+ entries there; keep existing PROMOTED/absorbed
      annotations as they are.
- [x] Fix tatr check findings best-effort, assuming recorded work was done
      properly where the record supports it:
      - closed-unchecked: tick Steps boxes whose close-out notes or landed
        commits evidence the work shipped; genuinely unshipped steps stay
        unticked and go on the residue list;
      - closed-not-approved: normalize nonstandard-but-approving verdict
        lines (e.g. "Verdict: APPROVE", "**APPROVE**") to
        "- VERDICT: APPROVE"; a review that really ended unapproved goes on
        the residue list untouched;
      - bad-severity: map to the nearest of BLOCKER/MAJOR/MINOR/NIT
        (LOW -> MINOR, NOTE/INFO/OBSERVATION -> NIT, FIXED -> the severity
        it had, keeping any "fixed in-review" note in the text), recording
        the mapping in the close-out.
- [x] AGENTS.md: add or refresh a "Development flow" section stating: /flow
      drives development here (plan/work/review/compound via tatr tasks,
      sprout worktrees, out-of-context round-1 reviews, DoD proofs with
      test:/cmd:/manual: notation); LESSONS.md at the repo root is the
      lessons ledger, read before starting any task; `tatr check` (plus
      `--ledger LESSONS.md`) is the conformance gate. Keep the section
      short; do not restructure the rest of the file.
- [x] Verify: tatr check exit 0 (or residue listed in the close-out),
      tatr check --ledger LESSONS.md: zero ledger findings (task residue keeps exit nonzero), and the repo's own check
      suite still green.

## Definition of Done

- LESSONS.md at the repo root, old docs/ path gone, no stale references
  (cmd: test -f LESSONS.md && test ! -f docs/LESSONS.md && ! grep -rn "docs/LESSONS" --include="*.md" --include="*.sh" .)
- tatr check clean or residue documented (cmd: /home/alex/personal/tatr/tatr check;
  manual: user reviews the residue list at the goal's Finish)
- ledger lints clean (cmd: /home/alex/personal/tatr/tatr check --ledger LESSONS.md)
- AGENTS.md names /flow and LESSONS.md (cmd: grep -n "flow\|LESSONS.md" AGENTS.md)

## Notes

- Use the tatr binary at /home/alex/personal/tatr/tatr (the installed one
  may predate the check subcommand).
- Preserve history honestly: normalizations keep meaning; ticks record
  verifiably shipped work only (linter-adoption cleanup, per the precedent
  in tatr's own 20260720-152503).

## Close-out

### What changed

- Ledger moved to the repo root via `git mv` (history preserved). Intro
  updated to describe the Pending promotions flow; added an (empty)
  "## Pending promotions (3+ occurrences, user decides)" section - the
  highest count in the ledger is x2, so nothing moved into it. The one
  RETIRED annotation (`separate-usage-reset-from-log-reset`) kept as is.
- Reference sweep: AGENTS.md (Layout table row, "Where records go" text) and
  the `scufris/agent.py` module docstring (a `.py` reference the DoD grep's
  --include set does not cover) now point at the root ledger. No README,
  script, or CI reference to the old path existed.
- AGENTS.md: new short "Development flow" section (/flow drives
  plan/work/review/compound; tatr tasks; sprout worktrees; out-of-context
  round-1 reviews; DoD proofs in test:/cmd:/manual: notation; root
  LESSONS.md read before any task; `tatr check` + `--ledger LESSONS.md` as
  the conformance gate). Rest of the file untouched.
- 16 closed-not-approved reviews fixed: each had only a "### Verdict"
  heading with "APPROVE. <prose>"; inserted a standard "- VERDICT: APPROVE"
  line under the heading, keeping the prose rationale.
- Severity normalization across REVIEW.md files: LOW -> MINOR (48
  occurrences, incl. verdict-prose references to "LOW items"), NOTE -> NIT
  (18 occurrences). No INFO/OBSERVATION found.
- FIXED-in-review findings mapped to a severity, keeping the note:
  - 20260720-102600 ("1 tools" pluralization) -> NIT (FIXED in-review)
  - 20260719-223111 (scrollTop regression introduced in review) -> MAJOR
    (FIXED in-review)
  - 20260719-223106 (shadowed `window` global, "was MAJOR-ish") -> MAJOR
    (FIXED in-review)
  - 20260720-102601 (tool catalog shown while tools disabled) -> MINOR
    (FIXED in-review)
  - 20260720-122517 (stray `import os` placement) -> NIT (FIXED in-review)
- 7 closed-unchecked tasks: ticked 29 Steps boxes evidenced by the tasks'
  own Implementation sections, their APPROVE reviews, and the landed code;
  5 boxes left unticked (residue below).

### Residue (left unticked / unfixed, user decides)

- 20260719-223102 step 5: "npm run ci green + a serve smoke: ... renders
  formatted (user-eyeballed) and the copy button works" - npm run ci green
  is recorded, but no record of the serve smoke, the eyeball, or the copy
  button working.
- 20260719-223103 step 6: "LIVE serve smoke ... verify against this host's
  codex" - the review explicitly notes a real codex turn was NOT run
  (fake-codex pipe only, "the live look is the user's eyeball").
- 20260719-235505 step 3: "sessions.py: ... DEBUG counts for list/read,
  INFO for delete" - shipped logging covers list + delete only;
  read_context/read_transcript/read_usage have no log calls.
- 20260720-002621 step 3: mostly shipped, but the "tool events append live
  chips" clause shipped as the "ran <tool>" status-line feed; tool CHIPS
  arrived later (20260720-122513). Ambiguous, left unticked.
- 20260720-002621 step 4: `.chat__thinking` shipped; the "token
  cursor/typing affordance" did not (no such CSS exists).
- DoD note: the "no stale references" grep matches only THIS task's own
  TASK.md (its step text and the DoD cmd line itself quote the old path) -
  self-referential, no real stale surface.

### Check results

- `tatr check`: 4 findings remain (closed-unchecked on 223102, 223103,
  235505, 002621) - exactly the residue above; was 23 findings before.
- `tatr check --ledger LESSONS.md`: zero ledger findings (the 4 task
  findings above are the only output).
- Python suite (nix develop): `ruff check .` and `ruff format --check .`
  pass; `pytest` 138 passed. `mypy .` fails with 18 errors in 2 test files
  (FakeAgent/LogRecord) - PRE-EXISTING: the identical 18 errors reproduce
  on the untouched master checkout at the same commit (4f99091). Not fixed
  here (docs-only task); needs its own task.
- Frontend: `npm run ci` green (format + lint + 5 vitest files + webpack
  build).
