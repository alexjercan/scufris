# Retro: Spike: choose the persistence mechanism, migration, and recovery policy

- TASK: 20260801-100405
- BRANCH: master (landed 6ca5240, 8948247, 27d6eb1; no sprout - the deliverable
  is two records plus a measurement harness, no `scufris/` or `tests/` file
  touched)
- REVIEW ROUNDS: 3

## What went well

Building the REJECTED candidate properly was the decision this spike turned on.
`JsonStore` in the harness is the incumbent discipline with every predecessor
constraint applied - per-store lock, unique temp name, `fsync` of file and
directory, non-tolerant loader, commit-or-revert - and it PASSES the headline
concurrency test 200/200 with zero exceptions. Had the comparison used the
shipping code, the record would have "proved" SQLite by beating a design nobody
was proposing, and the four measurements that actually decide it (multi-record
tearing, reader latency, cross-process loss, append growth) would have read as
piling on rather than as the argument.

Measuring beyond the axes the task listed paid for itself twice. `procs` was
not requested and produced the worst single result in the record - 150 of 300
writes lost with `raised=0`. `crash` was not requested and closed the
predecessor's own stated limitation ("no crash injection") with a tie, which is
a result the record is better for carrying.

The predecessor's evidence was consumed rather than restated. Each of its nine
constraints maps to a numbered rule in DECISION.md section 2, and its four open
questions are answered by name in section 3.

## What went wrong

**The isolation number was a harness artifact, not a measurement.**
`scenario_isolation` constructed 200 `SqliteStore` objects and closed none of
them, so setup 201 was timed with 200 connections open. It reported 28.963ms.
With a `close()` per iteration it reports ~10.4ms, stable across four runs. The
decision that seemed sound at the time: the scenario only needed to CREATE
stores, so teardown looked irrelevant to what it measured. It was not - a
setup-cost loop that leaks handles measures the leak. The record then quoted
that number as a cost the user accepted.

**An unreproducible number papered over it.** Noticing the figure looked high,
the response was an ad-hoc heredoc that gave 6.6ms and a sentence in SPIKE.md
attributing the gap to machine load. The sentence also mis-described its own
source as a seven-table measurement when the heredoc used one table. That is
two failures in one move: quoting a number no committed artifact produces, and
explaining away a discrepancy instead of finding its cause. The cause was two
lines of code.

**A single run produced a claimed verdict that reversed.** Fixing R1.4 -
correctly, that the record should name the axis where the rejected candidate
wins - the correction asserted retention as that axis from ONE run (5.26 vs
10.56ms). Three re-runs gave 5.19/4.91, 4.22/16.08 and 63.95/4.66: SQLite wins
two of four and the spread is an order of magnitude wider than the gap. The
ledger already says counts do not reproduce across runs. It did not say that
the ORDERING between two candidates is itself a count when the gap sits inside
the noise, and that is the form the mistake took.

**A decision invariant was stated at a scope no single task could satisfy.**
DECISION.md section 4 said the legacy import is "one entry point migrates the
WHOLE state directory as a single transaction" and that "partial migration is
not a state that can exist" - while this same spike had just written a Step
into 20260729-102147 importing `projects.json` alone. An implementer following
the decision literally would have had to break the pilot task's scope or
violate the decision. The claim appeared in four places and had to be corrected
in all four.

**The user accepted numbers that later changed.** Manual acceptance was taken
before review, quoting 28.963ms per test fixture and a retention claim that did
not survive. The direction and the mechanism never changed - the corrected
figure is ~3x cheaper and strengthens the case - so the acceptance stands, but
the ordering meant the user approved a record whose numbers the review then
moved.

## What to improve next time

Diagnosis, per the three questions:

- **Breadth.** The diff is two records plus one harness, large because ten
  scenarios were needed to cover six named axes plus the next epic's workload.
  No independently landable split existed: the harness is one artifact whose
  value is that both candidates run under identical conditions. Not a missed
  split.
- **Churn.** Three rounds, seven findings. R1.1 (the scope contradiction) is
  the one a plan-time question would have caught: the cold-reader rationale test
  in `plan/decision.md` asks whether a reader with no context could act on the
  record, and an implementer at 20260729-102147 is exactly that reader. R1.2 and
  R1.3 were not plan failures - they were verification failures inside this
  task, caught only because the review re-ran the harness and re-counted the
  suite instead of reading the prose. R2.1 and R2.2 were regressions from
  round-1 fixes, which is the strongest argument here for re-verifying a FIX at
  the same standard as the original claim.
- **Context.** No compaction, no handoff, no measured pressure. Review ran
  in-session because this session's rules prohibit subagent delegation; that
  exception is recorded in REVIEW.md and was compensated by re-deriving every
  cited code location, re-running the harness, and independently counting the
  test suite - which is what produced R1.3.

## Action items

- Ledger: `a-setup-cost-loop-must-release-what-it-allocates`,
  `an-ordering-between-candidates-is-a-count-not-a-verdict`,
  `state-a-decision-invariant-at-the-scope-one-task-can-satisfy`.
- Ledger bump: `pin-evidence-to-the-commit-that-produced-it` to x2 - the
  heredoc figure is the same family as the mis-pinned commit, evidence in the
  record that the committed instrument does not produce.
- Ledger bump: `dod-proof-must-exercise-the-named-claim` to x4 - the migration
  DoD grep (`rg -n "idempotent|backup|rollback|partial|corrupt|downgrade"`)
  returned 8 matches against a DECISION.md whose migration policy contradicted
  its own implementation lane. Already promoted; 20260801-104446 owns the guard,
  so this occurrence is evidence for it rather than a new decision.
- No follow-up task. The three implementation tasks were refined in place
  (task Step 7) and the epic index carries the decision and the accepted
  tradeoffs.
