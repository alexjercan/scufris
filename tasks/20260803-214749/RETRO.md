# Retro: Move the host control client into packages/hostctl

- TASK: 20260803-214749
- BRANCH: refactor/hostctl-package
- REVIEW ROUNDS: 3

## What went well

The plan called this "the one child that is NOT a pure move" and budgeted for
real edits, and that framing paid: `eventbus`, the generic `Supervisor` half,
the `Settings` narrowing and the two row classes each landed as their own green
commit before the package commit, so the carve itself was reviewable as a move.
Four preparatory commits, one carve commit, no rework between them.

Rounds 2 and 3 were cheap because the branch already had two derived guards -
`test_every_package_model_is_registered` and
`test_no_package_imports_a_sibling_private_module` - so the review had something
concrete to attack instead of prose to argue with.

## What went wrong

**The boundary rule shipped with a hole nobody needed (R1.1).** The
implementation had `env.py` import a sibling's private `models` module and paid
for it with a `SCHEMA_ASSEMBLY` exemption in the very test this task exists to
install. The premise seemed sound at the time - a package's row classes are
private, so a facade cannot expose them - and it is false for the wrong reason:
Alembic needs the import SIDE EFFECT, not the classes, and the facade already
produces it. The tell was available for the cost of one `python -c`: import the
facade, print `Base.metadata.tables`. Writing an exemption is a moment to run
that check, not to write a paragraph justifying it.

**A guard was claimed to hold rather than broken (R2.1).** Round 1's response to
R1.1 argued that `test_every_package_model_is_registered` survived the change
because it reads `env.py` with `ast`. True, and irrelevant: the test imported
`env.py`'s module list into an interpreter where `scufris_hostctl` was already
loaded, so it passed with the import deleted. Deleting the line and running the
full suite - 65 seconds - would have caught it in round 1 instead of round 2.

**Prose written against an overturned plan clause was not swept (R1.2).** The
plan's "no real socket" for the example was correctly reversed in DECISION.md
point 5, but the package README written in the same commit kept the old wording.
The decision record was updated; the surfaces written against the decision were
not.

The remaining round-1 findings (R1.3 through R1.7) were all one shape: stale
pre-move names, counts and task IDs that a `git mv` carries forward silently. A
move does not update the prose inside the moved file.

## What to improve next time

- **An exemption to a rule is a falsification prompt.** Before writing an
  allowlist entry, run the cheapest experiment that would make it unnecessary.
- **A guard's claim is only worth what breaking the guarded thing proves.**
  When a change touches what a test guards, delete the guarded thing, run the
  canonical gate, and paste the red. Reasoning about `ast` and import order is
  exactly the class of argument that is locally correct and globally wrong.
- **In-process import checks are vacuous under a full suite run.** Anything
  that asserts on module-level registration state has to run in a fresh
  interpreter, because `sys.modules` is shared across the whole session.
- **A reversed plan clause owes a prose sweep, not just a decision record.**
  Grep the clause's own wording across the diff after recording the reversal.
- **`git mv` moves the file, not the sentences inside it.** After a move, grep
  the moved tree for the old module path, the old `:class:` targets, and any
  count the move changed.

## Action items

- R3.1 stays open as a MINOR on an APPROVEd branch:
  `tests/test_db_migrations.py`'s subprocess helper uses `check=True` with
  `capture_output=True`, so a child that raises on import reports only a
  non-zero exit status. Worth folding into whichever task next touches that
  file; it does not justify a fourth round here.
- Task IDs in docstrings are repo-wide and pre-existing (R1.7's accepted
  scoping). If `AGENTS.md`'s rule is to mean anything, the sweep needs its own
  task; this branch fixed only the file the finding named.

## Diagnosis

**Breadth.** 76 files, +2603/-1145, and the size is inherent rather than a
missed split: the four preparatory hoists were each independently landable and
were in fact landed as separate commits, and what remains is one package's worth
of `git mv` plus its import re-pointing. The plan predicted this shape and it
held.

**Churn.** The from-scratch challenge in `plan` would not have caught R1.1;
`plan/decision.md`'s cold-reader rationale test would have. The exemption's
justification was written as a paragraph a cold reader would accept, and the
paragraph was wrong on a fact a one-line experiment settles. The rationale test
needs a companion: when the rationale is an empirical claim about what the code
does, run it.

**Context.** No context pressure observed. No checkpoint, no compaction warning,
no handoff. Review rounds 1-3 each ran in a fresh out-of-context reviewer, which
is what let round 2 find a defect round 1 had reasoned past.

## Landing message

```
refactor(hostctl): carve the host control client into packages/hostctl

Moves the unprivileged client that drives hostd - propose, preview, approve,
dispatch, watch, plus the NixOS configuration change flow and its generation
rollback - behind one distribution boundary, so the completed host-agency
pillar can be left alone while the rewrite happens beside it.

Not a pure move. EventBus and the generic half of Supervisor are hoisted into
scufris_core first, ConfigChangeService's dependency is narrowed from Settings
to two values, and two row classes leave scufris/db/models.py for the package
that owns them.

Two derived guards ship with it: core stays domain-free, and no workspace
member imports a sibling's private module. The migration environment imports
the scufris_hostctl facade rather than its models, so the second rule has no
exemptions, and its registration check runs in a fresh interpreter so a dropped
import fails the canonical suite.
```
