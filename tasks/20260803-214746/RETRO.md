# Retro: Bootstrap the uv workspace and the core package

- TASK: 20260803-214746
- BRANCH: refactor/uv-workspace-core
- REVIEW ROUNDS: 2

## What went well

- The understanding pass deleted four of the things the task originally said to
  move (`enums.py`, `ids`, `time`/errors, "the session factory") by checking
  each against the tree. Two of them did not exist at all. `core`'s declared
  dependencies dropped from five to one before a line was written, and the
  smallest possible first member is exactly what this task was for.
- The boundary tests read the SOURCE with `ast` instead of importing it, and
  `test_the_domain_free_check_rejects_the_pre_move_tree` points the same helper
  at `scufris/db`. The falsifier is a permanent test, not a one-shot
  demonstration, so the green arm cannot become green by checking nothing - and
  it caught its own regression the moment `Base` left `db` (the declarative walk
  seeded from `DeclarativeBase` alone found nothing).
- The allowlist over a property check was argued in the plan and confirmed in
  the writing: "declares no table" waves through `EventBus`, `Supervisor` and
  `RunPhase`, which is the junk drawer the test is named against. Adding
  `logsetup` to `core` then cost a written justification, which is the workflow
  working as designed.
- Two edits outside the Steps - ruff's `known-first-party` and `flake.nix`'s
  `export REPO_ROOT=$PWD` - were disclosed under Difficulties rather than folded
  in. The `REPO_ROOT` one exposed a latent hole in the check sandbox that any
  package layout would have hit.
- Breadth: 53 files, but most are one-line import re-points from a mechanical
  `git mv` census. No independently landable split was missed - the workspace
  cannot exist without a member, and the member cannot exist without its
  importers moving in the same commit.

## What went wrong

- The DoD's build proof covered the uv2nix path only. Nothing in the proof set
  touched `uv build`, and R1.1 (BLOCKER) lived exactly in that gap: creating a
  second distribution made the root wheel carry `Requires-Dist: scufris-core`
  while `uv build` still produced only the root wheel, so the published artifact
  was uninstallable. The release job's own smoke test could not have caught it -
  it installed one wheel chosen by `ls`.
- Why it looked sound: `nix flake check && nix build .#scufris` is this repo's
  habitual "the build is unchanged" proof and it passed. It is a complete proof
  for a task that changes code inside one distribution, and this was the first
  task in the repo's history to change the NUMBER of distributions. The proof
  set was inherited from tasks where it was sufficient.
- Churn: the plan's from-scratch challenge asked whether `core` was the right
  cut and what belonged in it. It never asked what the diff changes about the
  RELEASE. One question - "which artifact does CI publish, and does any proof
  here look at it?" - would have produced the `uv build --all-packages` step at
  plan time instead of in a blocker.
- The Step census drifted twice, both times on a hand-made count: "18 modules
  plus 25+ test files" (corrected in planning; it was a count of `scufris.db`
  importers, not `engine` ones) and "twelve logsetup importers" (R1.11 - it is
  eleven; four are function-local imports inside MCP server `main()`s, which is
  why an eyeball count missed the shape).
- Context: no threshold crossing, compaction warning or handoff was recorded.
  One confusing red suite was spent before the `uv lock` -> re-enter
  `nix develop` ordering was understood.

## What to improve next time

- Paste the `rg -l` output into the Step. A list is checkable against the tree
  and a number is not. Both wrong counts here were typed by hand; the round-2
  re-derivation matched for the first time precisely because the record had
  stopped carrying a number.
- When a task changes the SHAPE of what ships - a new distribution, a new entry
  point, a new artifact - the DoD needs a proof that runs the release path, not
  only the dev-build path. `nix flake check` is not a release proof.
- `uv lock`, then re-enter `nix develop`, BEFORE running anything. The dev venv
  is a derivation built from the lock. Now a line in `AGENTS.md`.
- Check `flake.nix` early in each of the four remaining carves. It had to change
  here for the move to be provable at the gate at all, and that is a property of
  adding a member, not of this member.

## Action items

- R2.1 (NIT) stays open and non-blocking: `docs/RELEASING.md` step 1 tells the
  operator to bump the root and every member pyproject but never to re-run
  `uv lock`, so a bumped-but-unlocked tree passes `release_tools check` and
  hands the nix build the old version. One line, but it is a release-procedure
  edit arriving after an APPROVE; it belongs with the next carve
  (20260803-214747), which touches member versions again, rather than in an
  unreviewed amendment here.
- The branch is one commit behind `master` (`b2ebfcf`, which re-briefed the
  carve children and re-ordered the sprint). That commit edits the four sibling
  task records this branch also edited. Rebase before landing.
- The `manual:` check in the parent epic 20260803-213242 (Manual Acceptance) is
  still `(pending)`: the maintainer names the owning package for a given
  concern. Correctly not self-ticked; it does not block this task.
- `check_agreement`'s member arm and the `--all-packages` smoke step already
  cover the four remaining carves. No follow-up task needed there.

## Landing message

```
refactor(packaging): bootstrap the uv workspace and the core package

Make the repository a uv workspace with one member, packages/core ->
scufris_core, holding engine.py (moved whole), the shared Base out of
db/models.py, and logsetup.py. Its __init__ is the entire public surface -
eleven names with an explicit __all__ - and sqlalchemy is its only dependency.
The thirteen row classes stay at the root, and scufris/db keeps re-exporting
the four composition names, so the modules importing from scufris.db are
untouched.

tests/test_package_boundaries.py replaces two README claims with ast checks
over the source tree: an allowlist for what core may contain, plus the same
helper asserted to FAIL against the pre-move scufris/db.
examples/core_unit_of_work.py is the runnable proof, gated by
tests/test_examples.py off an explicit offline opt-in list.

Splitting one distribution into two changes the release: the root wheel now
declares Requires-Dist: scufris-core, so the release job builds
--all-packages and smoke-installs the whole set in one resolution, and
release_tools.check_agreement fails when a member's version differs from the
root's.

The epic's decisions are recorded in tasks/20260803-213242/DECISION.md.
```
