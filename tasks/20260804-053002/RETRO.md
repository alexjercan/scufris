# Retro: Prove the declared dependency graph and the example gate

- TASK: 20260804-053002
- BRANCH: test/prove-declared-graph-and-example-gate
- REVIEW ROUNDS: 3

## What went well

The plan was executable as written and the diff never grew past it: two test
files, three doc surfaces, no production code, exactly as the Story scoped. The
graph edges the plan predicted were the edges on disk, so the green test was
green on first run.

Every round's rework made the change stronger rather than merely compliant.
Round 1 turned four asserted-in-prose checker arms into driven ones; round 2
deleted an optimisation that had silently changed behaviour; round 3 corrected
two docstrings that claimed more than the code delivered. The mutation loop that
round 1 asked for became the standing evidence format for rounds 2, 3 and 4, and
the round-3 reviewer reproduced every recorded number exactly.

Out-of-context review earned its cost three times over. Every finding in every
round came from a reviewer with no sight of the implementing session, and the
two that mattered most - R1.1 and R2.1 - were both cases where the implementing
session had a reason to believe a thing was proven and no test said so.

## What went wrong

The root failure repeats once per round, in the same shape: **a claim written in
prose beside code, where the code could have carried it.**

- R1.1 and R1.2: `_graph_problems` had four arms; the falsifier pinned one
  through the checker and one through a helper, and the close-out said "the
  falsifier was proven to bite" - a true sentence that read as coverage of all
  four. Deleting three arms left the suite green.
- R2.1: the `done` memo taken for a NIT changed what `_cycles` RETURNS, dropping
  cycles on overlapping-node graphs, and the docstring paragraph justifying it
  asserted the opposite. Nothing could fail.
- R3.2: `_cycles` claimed to deduplicate per set of mutually reachable nodes,
  which its own falsifier contradicts.

The failed decision worth naming is R1.7's memo. It seemed sound because it was
filed as a NIT, its diff was three lines, and the reasoning about DFS memoisation
is correct for *existence* detection - which is what the caller needs. What was
missed is that the function's contract is *enumeration*, and the change was
therefore behavioural, not cosmetic. A NIT that alters a return value is not a
NIT.

Process cost: one self-inflicted detour. The first mutation harness restored
mutated files with `git checkout -- <path>`, which destroyed the uncommitted
fixes it was testing and reported four arms falsely GREEN. The tell was that only
the first mutation in the list ever came back red.

## What to improve next time

- A checker returning a list of messages gets a mutation loop per arm, written
  as a scratch script, and its per-arm results pasted into Evidence. Not "the
  falsifier bites" - which arm, which mutation, which result.
- Mutation harnesses snapshot file text in memory. Never `git checkout` a file
  that holds unstaged work.
- Apply the branch's own standard to its docstrings. In a change arguing that an
  undriven claim is worth nothing, three of the four review findings that
  mattered were undriven claims in comments.
- Treat any finding that changes a return value as at least MINOR regardless of
  the severity it arrives under, and pin the property it touches before taking
  it.

## Action items

- Filed 20260804-101727: `uv run pytest tests/` fails collection with
  `ModuleNotFoundError: No module named 'test_host_actions'` from
  `tests/conftest.py`. Pre-existing, found during round-1 review, not this
  branch's problem.
- No follow-up work is owed on this branch. All ten findings across three rounds
  are ticked.

## Diagnosis

**Breadth.** The diff is small and stayed small: 175 lines added to
`test_package_boundaries.py` against 8 removed, 111 added to
`test_examples.py`, and 6 across two doc surfaces. No split was missed. The one piece of scope found late -
`tests/test_examples.py` having the same unasserted-arm defect the findings only
charged against its sibling - was correctly absorbed here rather than deferred,
because it was the same defect in the same change.

**Churn.** Three rounds of rework trace to one plan-time gap. The Steps named
`_graph_problems` "with its four arms" and named the falsifier, but never said
that each arm must be driven independently - so a falsifier that exercised the
function satisfied the Step's literal text while proving a quarter of it. The
question that would have prevented it is the proof-shape one: for a checker with
N arms, the DoD needs N pieces of evidence, not one command that exits 0. The
plan is the subject here; the same gap let a NIT-sized behavioural change through
two rounds later.

**Context.** No context pressure was observed or recorded: no checkpoint, no
compaction warning, no handoff. The three out-of-context review rounds were
delegations by design rather than by pressure. Nothing to split, defer or load
later on this evidence.

## Landing message

```
test: prove the declared dependency graph and the example gate

The epic's dependency direction lived only in README prose, and the example
gate was a hand-written opt-in tuple with no relation to the member list, so a
new `core -> host` edge or a package that never shipped an example both landed
green.

`DECLARED_GRAPH` in tests/test_package_boundaries.py is that direction as a
literal, checked for EQUALITY against the sibling imports the tree really makes
and for acyclicity. `EXAMPLES_BY_MEMBER` in tests/test_examples.py requires
every workspace member to name an example that is on OFFLINE, exists, and
imports it. Both checkers report every failing arm rather than the first, and
each arm is driven by a falsifier over hand-built inputs: deleting any one of
the ten arms, or the cycle detection they share, turns the suite red.

Tests only; no production code moves. AGENTS.md and scufris/README.md gain what
carving a new member now costs.
```
