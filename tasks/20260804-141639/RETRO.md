# Retro: Close the three open round-2 findings on the delivery contract

- TASK: 20260804-141639
- BRANCH: feature/delivery-contract-round2
- REVIEW ROUNDS: 1

## What went well

Test-first held. The new case was written to the exact `LookupError` the plan
predicted, red on base at `store.py:313`, and the recording pass re-derived that
red independently by restoring `master`'s `store.py` in the worktree rather than
trusting the close-out. `test_delivery_requires_its_event` was left byte-
identical, which is what keeps the narrowed raise honest - the contract got
looser in exactly one direction and a test still pins the other.

The plan predicted its own README line numbers were stale and said so. The build
answered with a width scan over the whole file instead of trusting either the
cited lines or the reviewer's memory, and found the real second orphan at 393.
Cheap defence, right instinct.

APPROVE at round 1, four findings, all MINOR or NIT.

## What went wrong

Steps and the Definition of Done disagreed about the same artifact. Step 5 said
`tasks/20260804-115319/DECISION.md` section 6 "is history and stays as it is";
the DoD's side-by-side criterion required that record to carry no remaining
claim that a correct caller cannot reach the raise. NOTES.md had already
proposed the reconciling third option - a dated round-2 note rather than a
silent rewrite - and the plan dropped it. The implementer arbitrated at build
time and disclosed the conflict in the close-out rather than ticking it, which
is the right call under a bad instruction, but both review lanes raised the
unmet criterion independently and it is still open as R1.2.

Collapsing `claim_delivery` to an unconditional `return True` (Step 3) narrowed
a guarantee the first docstring paragraph still states unqualified: `False` only
for a `confirmed` row. Under a deferred begin, a conflict loser now answers
`True` for a row that may already be `confirmed`. Unreachable in-tree and the
package prefers a duplicate to a loss, so it is prose precision, not behavior -
but the Step named the paragraph to change and the neighbouring paragraph making
the same promise was not swept. That is R1.1.

## What to improve next time

Churn: this whole task is rework on `20260804-115319`, and its root cause was an
unreachability claim written into a decision record with no test behind it -
"no correct caller reaches that, because a `True` claim always leaves the row
`claimed`". The record's own earlier half contained the counterexample. The
plan-time question that would have caught it is not the from-scratch challenge
but a narrower one: when a docstring or decision asserts a branch is
unreachable, write the test that tries to reach it, or state the assumption the
unreachability rests on.

Breadth: the diff is small and cohesive - one function pair, one test, one
README section - and needed no split.

Process: a plan gate should reject a Step and a DoD bullet that give opposite
instructions for one file. The conflict was visible in the plan text and in
NOTES.md before any code was written.

Context: no pressure observed. Review was delegated to two out-of-context lanes
(correctness/concurrency, spec/design/docs); they agreed on the section 6
finding and split cleanly otherwise, which is the shape that justifies lanes on
a concurrency-and-contract diff.

## Action items

- `tasks/20260804-184521`: follow-up for the two open MINORs - qualify
  `claim_delivery`'s completed-delivery promise with the immediate-begin
  assumption (R1.1), and
  append the dated round-2 note retiring section 6's falsified clause on
  `tasks/20260804-115319/DECISION.md` (R1.2). Lane 2's channel author reads both
  surfaces, so they should be true before that lane starts.
- The two NITs (R1.3 collapse and re-aim the confirm guard, R1.4 monkeypatch the
  `confirmed_at` clock) are take-it-or-leave-it and carry no follow-up.

## Landing message

```
fix(chat): make confirm the mirror of claim and give claim one write

`confirm_delivery` raised whenever no `claimed` row matched, but
`claim_delivery` hands an abandoned `claimed` row back to a second pass, so
two overlapping passes over one channel both send and the second confirm
raised into an otherwise-correct caller loop. Confirm now re-reads and raises
on exactly one input, a confirm for a key that was never claimed; an
already-`confirmed` row is a no-op that keeps the first confirmation's
`confirmed_at`, which the `claimed` guard on the UPDATE enforces.

`claim_delivery` drops its conflict-loser re-SELECT and is one read and one
write. The docstrings and README section 5 state what the code does rather
than an unreachability the tests falsify.
```
