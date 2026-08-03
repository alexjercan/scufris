# Review: Make the config-change restart proofs reopen the database and cover the reap bound

- TASK: 20260803-014401
- BRANCH: test/config-change-restart-reap-proofs

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tasks/20260803-014401/DECISION.md:66 - the record says
  "Modelling a real crash is a follow-up, seeded from this record", and no such
  task exists: nothing under `tasks/` names it and the parent epic's frontier
  does not list it. The gap therefore lives only in an append-only record no
  tracker surfaces. Create the follow-up task (re-establish a `building` row
  through `ConfigChangeStore` after a clean shutdown, per the second
  alternative), name its id in DECISION.md's Consequences, and amend the DoD
  clause at `tasks/20260803-014401/TASK.md:109` to say one of the two restart
  proofs, so the criterion matches what shipped.
  - Response: fixed in 5f9ca49. Filed 20260803-113000 "Prove the startup sweep
    clears a building row orphaned by a crash" (p35, 20260729-102145, the second
    alternative as its preferred shape), added it to the epic's Lane B child
    list so the frontier surfaces it, named it in DECISION.md's Consequences,
    and rewrote the DoD clause to name
    `test_a_configuration_change_survives_a_restart` alone. Step 2 also carries
    an explicit NOT DONE note so no reader takes the empty checkbox for an
    oversight.
- [x] R1.2 (MINOR) tests/test_nixos_config_change.py:626 -
  `test_a_build_interrupted_by_a_restart_does_not_block_the_repo`'s docstring
  says it proves a build "killed by a restart" is swept, which DECISION.md 1
  establishes is false: the row is `building` at the restarted app's startup
  only because the first app is still alive and still hanging. The test is
  byte-identical to the base, but this branch is what discovered the overclaim,
  so a reader who reaches it from here re-derives the wrong belief. Add a
  comment above `client = make_client(` at line 635 saying the first process is
  deliberately left open because a graceful shutdown writes `cancelled`, not
  `building` - see 20260803-014401 DECISION.md 1.
  - Response: fixed in 5f9ca49. A four-line comment now sits above that
    `make_client` call, citing DECISION.md 1 and the follow-up 20260803-113000,
    and naming what the test does prove: the live-process case.
- [x] R1.3 (NIT) tests/test_nixos_config_change.py:587 - the eight comment lines
  restate NOTES.md and DECISION.md prose, against the repo rule that
  explanatory prose belongs in task records. Cut 587-594 to the two
  load-bearing facts: `create_app` memoizes the handle process-wide, so the
  first client must exit - and hence cannot be a `make_client` one, whose
  ExitStack unwinds at teardown - before the restarted app is built.
  - Response: fixed in 5f9ca49. The eight lines are now three, carrying exactly
    those two facts.

- Process signal: the plan's Step 3 escape hatch fired as designed and the
  finding is well recorded, but it left a DoD clause half-satisfied and the
  follow-up it promised unfiled. The split the discovery implies - a proof that
  models a process dying without shutdown hooks - is the missing task.

Verified in-session: `ruff check .`, `ruff format --check .`, `mypy .` clean;
full `python -m pytest` exit 0; `tatr check` exit 0. Re-derived independently:
making `_reap` return immediately turns `test_the_change_registry_stays_bounded`
red, so the new bound assertion is load-bearing, and the settled row sits second
of four so a seq-only reap fails the first assertion. Confirmed no follow-up
task exists by grepping `tasks/` and reading the parent epic's frontier - the
one record that cites this id, 20260803-022018, is about record lint. No
existing assertion was weakened or deleted; the only production change is the
`abandon_builds` return type, whose sole caller already discarded it, and it
appears in no doc surface outside `tasks/`.

Not verified: the "red on the base" halves of the first two DoD proofs by
checking out master; the equivalent was confirmed by mutation instead.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

No findings. All three round-1 responses verified against disk, and the fixes
introduced no regressions: 5f9ca49 touches comments and task records only, so
the branch's sole production change remains `abandon_builds(self) -> None`.

- R1.1 confirmed: `tasks/20260803-113000/TASK.md` exists (p35, PARENT
  20260729-102145, preferred shape = re-establish a `building` row through
  `ConfigChangeStore` after a clean shutdown, which is DECISION.md 1's second
  alternative), is listed in the epic's Lane B at
  `tasks/20260729-102145/TASK.md:68`, is named in DECISION.md's Consequences,
  and the DoD clause now names `test_a_configuration_change_survives_a_restart`
  alone. Step 2 carries an explicit NOT DONE note, so the empty checkbox reads
  as a finding rather than an oversight. Its Story-plus-Notes shape matches the
  repo's other unplanned backlog tasks and `tatr check` is clean.
- R1.2 confirmed: `tests/test_nixos_config_change.py:630-633` carries the
  comment above the `make_client(` call, citing DECISION.md 1 and
  20260803-113000 and naming the live-process case as what the test does prove.
  Its claim holds - the cancellation handler in `scufris/hostconfig/changes.py`
  writes `CANCELLED` before re-raising.
- R1.3 confirmed: the restart comment is three lines
  (`tests/test_nixos_config_change.py:587-589`) carrying the memoized handle
  and the `make_client` ExitStack facts; the ExitStack claim checks out at
  `tests/conftest.py:358`.

Re-derived independently, and this is the round's load-bearing check: moving
`second = _app(...)` back inside the first client's `with` block turns
`test_a_configuration_change_survives_a_restart` red on
`assert second.state.db is not first.state.db`, with both sides printing the
same `Database` object. The identity assertion therefore discriminates the
shared-handle case rather than passing by construction. The worktree was
restored to HEAD afterwards; `git status` clean.

Verified in-session: `ruff check .`, `ruff format --check .`, `mypy .` (228
files, no issues) clean; full `python -m pytest` exit 0;
`python -m pytest tests/test_nixos_config_change.py` exit 0; `tatr check` exit
0. DoD proofs run: `-k restart` and `-k stays_bounded` both exit 0, and
`grep -n "def abandon_builds(self) -> None" scufris/hostconfig/changes.py`
hits. Doc sweep: `abandon_builds` appears outside `tasks/` only at its
definition and `scufris/app.py:423`, which already discarded the value.

Not verified: the "red on the base" halves by checking out master. The
equivalent was established by the mutation above and by round 1's `_reap`
early-return check.

No pending `manual:` proofs.

Process signal:
- Round 2 needed no code changes; the round-1 responses were accurate down to
  the line-level claims.
- `tasks/20260803-113000/TASK.md:19` cites
  `tests/test_nixos_config_change.py:619` for a test whose `def` is at 614 -
  inside the same test, harmless, and `tasks/` is append-only history.
