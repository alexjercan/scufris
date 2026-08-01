# Retro: Split the oversized test suites under the size cap

- TASK: 20260731-171432
- BRANCH: refactor/split-test-suites
- REVIEW ROUNDS: 1 (APPROVE, 2 MINORs + 1 NIT, all open)

## What went well

**The rig was written before the first edit and it is what carried the review.**
The move proof (normalized code-line multiset per commit) and the test-NAME set
difference were both throwaway scripts, both run on every one of the eight
commits, and both reported per commit in a table rather than as prose. An
out-of-context reviewer rebuilt an independent rig and reproduced all seven
per-commit numbers exactly - 7/16, 12/19, 6/24, 2/29, 14/65, 10/36, 0/34 - and
the in-session pass re-derived the 896-name set equality separately again.
Round 1 found no correctness defect. That is the whole return on the rig.

**A count would not have been enough, and the plan knew it.** The Step said so
explicitly ("the count alone is not the proof"), and the frontend half proved
it: 258 leaf `it` names identical, but two of them changed their `describe`
parent in C7. A count check reports green there; a name-set check reports the
move, which is how it reached the close-out as a disclosed difference instead
of a silent one.

**Splitting by BEHAVIOR held against the structure the epic had just built.**
Five source children had turned every module under these tests into a package,
which made the package mirror the obvious rule. DECISION.md refuted it by `ast`
attribution on three files before any code moved, and nothing at work time
contradicted it. The two places where a submodule DOES own a contiguous
behavior block (`host/thermal.py`, `telegram/render.py`) were taken as their own
files - the rule was "measure", not "ignore the package".

**The conftest boundary was decided by consumer count, not by convenience.**
Cross-domain setup moved (`PASSWORD`, `ORIGIN`, `SECRET`, `_settings`, `_login`,
`_propose` - six modules import them now, and lifting them deleted three
cross-test-module imports); domain-local setup stayed and siblings import it
(the Telegram bot harness, the host-inspection result factories). One rule, both
answers, recorded in advance.

**One commit per file, each deleting exactly its own ALLOWLIST entry.** That is
what makes `git rebase master --exec` a real proof rather than a tip check, and
it was green across all 10 commits with the tip hash unchanged.

## What went wrong

**A `docs:` commit carried a whole-file `ruff format` into `scufris/app.py`,
which the task Notes name as out of scope.** The intended edit was one docstring
citation. `master`'s `app.py` was never format-clean - the flake gate is
`ruff check` lint-only - so a correctly file-scoped `ruff format scufris/app.py`
still produced ~13 unrelated re-wrap hunks in the security-critical middleware
file, and 20260729-103712 now rebases its `app.py` split across a reformat it
did not ask for. The existing rule ("scope every format to the files you
edited") was followed and did not help: `ruff format` has no hunk scope. Caught
in review as R1.1, behavior-preserving, not fixed on this branch.

**The citation sweep followed the files the commit touched, not the files the
commit renamed away from.** The auth citations were repointed properly
(`scufris/auth/policy.py`, `scufris/README.md`, `examples/auth_session.py`,
`app.py`) because the auth split was the one being read at the time. The
identical sentence in `tests/test_host_mcp_server.py` - "pinned against captured
fixtures in `test_host_inspection.py`", stated twice - was missed, because that
file is not in scope and the sweep was driven by the diff rather than by the set
of names that stopped existing. R1.3.

**A comment that enumerates its consumers went stale the moment a consumer was
added.** `tests/conftest.py:181` names two importers of `_Helper`; six import it
now. The enumeration was accurate when written and is a maintenance liability by
construction. R1.2.

## What to improve next time

**Breadth.** The diff is large (32 files, 8 base files becoming 22) and that is
inherent, not a missed split: each base file is independently landable, each got
its own commit, and the commit sequence is forced only once (C2 after C1, for
the helpers C1 lifts). The one avoidable growth is R1.1's reformat, which is
scope the plan explicitly excluded.

**Churn.** Zero review rework - no finding above MINOR, no rounds beyond the
first. No plan-time question would have changed the outcome, with one exception:
a Step clause reading "revert format-only hunks in files outside scope" would
have prevented R1.1. The `ruff format` Step in this plan was written from the
prettier failures of 20260731-171431 and correctly anticipated the gate risk; it
did not anticipate the out-of-scope-file risk, which is a different failure of
the same tool.

**Context.** No context pressure is recorded for this task - no compaction
warning, no handoff, no checkpoint in the records. The plan sized the task
around the cap arithmetic in the epic (a 600-line Python file is ~8-9k tokens)
and eight files at one commit each stayed inside it. Nothing to split, delegate
or defer on this evidence.

## Action items

- R1.1, R1.2 and R1.3 are open MINOR/NIT findings, non-blocking. The two comment
  repoints (R1.2, R1.3) are one-line edits and can ride the next task that
  touches those files. R1.1 is deliberately NOT reverted here: the reformat is
  behavior-preserving and already committed, and unpicking it would rewrite a
  reviewed branch for a cosmetic gain.
- Ledger: `format-only-the-files-you-edited-not-whole-dirs` gains this task's ID
  and the new clause; `prove-a-move-only-refactor-with-a-normalized-line-diff`
  and `grep-the-private-names-a-split-moves-not-only-the-module-path` both bump
  to x2; `a-citation-sweep-follows-the-renamed-name-not-the-edited-file` is new.
- 20260731-233221 owns turning the pending promotions into repository guards.
  Nothing from this retro folds into this task.
