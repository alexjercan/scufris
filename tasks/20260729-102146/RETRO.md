# Retro: Spike: inventory app-owned mutable state and reproduce the write races

- TASK: 20260729-102146
- BRANCH: master (no sprout - the deliverable is a record plus its evidence
  script; no `scufris/` or `tests/` file is touched)
- REVIEW ROUNDS: 2

## What went well

The decision to reproduce rather than describe paid for itself twice over. The
spike's brief was written because a project audit had already DESCRIBED these
races; running them surfaced two facts no reading had produced - host proposals
are not persisted at all, and auth sessions already hold a lock across their
read-modify-write - and both changed the epic's Done Means 4.

The unique-temp-name control earned its place. Without it the record would have
recommended "give each writer its own temp file" as the fix. The control raised
nothing and still regressed the published file, which is the single most
load-bearing observation the successor gets.

Splitting the mechanism decision into 20260801-100405 held. Nothing in this
record chooses a mechanism, so the successor argues against measurements rather
than against a conclusion already drawn.

## What went wrong

Step 5 was ticked without being done. The DoD proof was
`rg -n "scufris/.*\.py:[0-9]+" SPIKE.md`, and the inventory table - a different
section, written for a different Step - satisfies that pattern on its own. The
proof came back green, so the checklist was ticked on the tool's report rather
than on the artifact. The read-modify-write enumeration, which is the output the
successor most needs, did not exist until review round 2 forced it. The decision
seemed sound at plan time because the proof named the right regex; what it never
named was WHICH section had to contain the match.

Round 1's own numbers were wrong twice, in the same way the thing they were
criticising was wrong. R1.1 correctly said the headline counts did not isolate
the claim, then offered a replacement ("45 of 110 finished agents lack an
outcome") measured by treating "create succeeded" as "finished". R1.6 correctly
said the evidence needed a commit, then pinned it to the commit the review was
written at rather than the commit containing the instrumented script - so the
block still was not re-derivable, which was the entire point of the finding.
Both were caught in round 2 only because the fixes were instrumented and re-run,
not because anything was read more carefully.

`nix flake check` was never run in round 1. `mypy .` covers the whole repo
including `tasks/`, so the evidence script's three `_report` type errors were a
red gate from the moment it was committed. The task's DoD lists `tatr check` but
no repo gate, because the task was framed as record-only - and the record turned
out to contain Python.

Both review rounds were self-reviewed. The `review` skill defaults to an
out-of-context reviewer for round 1 and this session's operator rules prohibit
subagent delegation, so the exception was recorded in each round's `- REVIEWER:`
line and compensated by re-deriving load-bearing claims from source and
re-running the reproduction. The compensations worked, but they worked by
re-running things. A spike whose entire value is that its evidence survives
scrutiny is the worst case for a self-review, and two findings that were
themselves wrong is the evidence for that.

## What to improve next time

Write the DoD proof against the SECTION, not just the pattern. A presence-grep
that any part of the document can satisfy is not a proof that a specific part
exists. The fixed version - a heading grep plus five locations that appear
nowhere else - could not have been satisfied by the inventory table.

When a review finding supplies a replacement number, instrument it rather than
quote it. The number in a finding is exactly as unverified as the number it
replaces; it arrived by the same route.

Pin evidence to the commit that contains the INSTRUMENT, which means committing
the script first and re-running against the committed tree. A commit chosen
because it is the current HEAD is a timestamp, not a citation.

Run the repo gate on any task that adds a runnable file, even one filed as
record-only. The gate's scope is the repo, not the directory the task thinks it
lives in.

## Diagnose

Breadth: not applicable in the usual sense - the diff is one record, one script
and the epic's Decisions bullet. It grew across rounds rather than at once, and
the growth was the missing Step 5 output plus two observations the review found.
No split was missed; the mechanism decision was already split out.

Churn: the plan-time question that would have prevented both rounds of rework is
the from-scratch challenge applied to the PROOFS rather than to the design -
"could this command pass while the Step is undone?". For Step 5 the answer was
yes and nobody asked. This is a plan-skill question, not a worker error: the
proofs were written before the evidence existed, which is normal, so they have
to be written to be falsifiable by the wrong section.

Context: one compaction occurred during this task, between round 1's commit and
the round-1 fix work. It cost nothing recoverable - the records carried the
state, which is what they are for - and no handoff or delegation was needed. No
threshold crossing beyond that is recorded, so nothing to split or defer next
time on context grounds.

## Action items

- Ledger: bump `dod-proof-must-exercise-the-named-claim` to x3 with the
  wrong-section instance; it reaches Pending promotions.
- Ledger: new `pin-evidence-to-the-commit-that-produced-it`.
- Ledger: new `a-review-findings-replacement-number-is-unverified`.
- Ledger: new `mypy-covers-tasks-dir-scripts-too`.
- Ledger: new `closing-a-child-does-not-tick-its-epic-row`. The epic's child row
  was left `- [ ]` after this task closed; the user caught it, no check did.
- No follow-up task. The successor 20260801-100405 already exists and consumes
  this record; the Recommendation's nine constraints are its input.
