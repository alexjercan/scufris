# Retro: Render agent reports as attributed quotations

- TASK: 20260804-115321
- BRANCH: feature/agent-report-quotations
- REVIEW ROUNDS: 2

## What went well

The sabotage pass inside the work phase caught a hole no reading would have:
`authorize`'s conversation scoping was green under a test whose only
cross-conversation case used an `event_seq` that existed in neither thread, so a
query with no conversation predicate at all still raised `LookupError`. The fix
went into the fixture, not the assertion. Every claim in the close-out's
evidence table reproduced exactly when the review re-ran it, which is what made
the honesty dimension cheap to clear.

Round 1 went to a reviewer with no sight of the implementing context, and the
one finding that mattered was one only a fresh reader would look for: it probed
`dataclasses.replace` against a claim the implementing context had already
convinced itself of.

## What went wrong

**The witness was specified by its mechanism, not by its guarantee.** The Step
said "whose `__init__` takes a module-private witness". That was implemented
literally and correctly, and the property it existed to buy was still open:
`dataclasses.replace` copies the existing instance's witness through, so a
holder of one legitimate decision re-targeted it at another conversation, event
and actor with a stdlib one-liner naming nothing private. The DoD, `DECISION.md`
and the README all asserted that could not happen.

Why the mechanism wording looked sound at plan time: `DECISION.md` had argued
the capability shape against three alternatives, and the witness was the
concrete artifact that argument produced. Naming the artifact felt like the
precise instruction. What it lost was the adversarial half - a frozen dataclass
has more construction routes than its `__init__`, and the Step never asked which
ones.

**The `CHANGELOG.md` entry was missed.** `AGENTS.md` requires one for a notable
change and both sibling lanes of this epic wrote one. The Steps enumerated five
concrete deliverables and did not name it, and the enumeration was followed
exactly. A checklist that is nearly complete reads as complete.

## What to improve next time

- **State a property as what an attacker cannot do, not as the mechanism that
  stops them.** "`OperatorDecision` cannot be minted outside `authorize`, by any
  route that does not name a private symbol" would have forced the `replace`
  question during implementation. "Its `__init__` takes a witness" did not.
- **Before writing "cannot" into a README or a DoD, enumerate the routes.** For
  a Python value type that is: the constructor, `dataclasses.replace`,
  `copy.copy`/`copy.replace`, pickle, and `object.__new__`. Decide for each
  whether it is closed or accepted, and let the prose say which. The final
  wording landed as a Round 2 NIT precisely because the headline asserted an
  absolute the body then withdrew.
- **Repository-wide rules are not the Steps' to remember.** `CHANGELOG.md` and
  the doc-surface sweep apply whether or not a Step names them; check
  `AGENTS.md` against the diff before handing back, not the Steps against the
  diff.

## Diagnose

**Breadth.** The diff is small and matches its plan: one new module, one test
module, three doc surfaces. No split was missed. The one structural deviation -
the README section landing as `### 3.1` under the actor rather than as a new
numbered section - was argued in the close-out and confirmed sound in review,
because a new section would have renumbered the surface table and section 8,
which this task's own `DECISION.md` cites by number.

**Churn.** Two review rounds, both traceable to the plan rather than to the
work. `plan`'s from-scratch challenge would not have caught either; what would
have is the cold-reader rationale test in `plan/decision.md` applied to the
DoD's own wording - a cold reader asked "cannot be constructed without the
module-private witness: how would I try?" produces the `replace` route in one
step. The `CHANGELOG.md` miss is the same class one level up: a Step list that
enumerates deliverables silently claims to be exhaustive over the standing
rules too.

**Context.** No context pressure observed - no checkpoint, no compaction
warning, no handoff. One process cost worth recording: the first Round 2
reviewer completed every verification and then looped for several minutes
trying to extract a summary line from pytest before being stopped, and its one
open question went to a second reviewer with a narrow brief. Recorded in
`REVIEW.md` as a process signal. Next time, tell a review subagent which checks
the recording pass has already run, so it spends its budget on judgement rather
than on re-deriving a green suite.

## Action items

- None requiring a task. Both improvements above are plan-authoring habits, and
  the R2.1 NIT is a one-line README wording change that Lane 2 will touch when
  it relocates the type to `scufris_core`.

## Landing message

```
feat(chat): mint an operator decision only a committed operator event can

scufris_chat.authorize re-reads one committed event under the caller's
open connection and mints an OperatorDecision from it, refusing every
non-operator actor by name. A stop gate takes the decision as an
argument, so a caller holding none cannot phrase the call: "an agent
report is data, never an instruction" becomes a property rather than a
comparison every call site has to remember.

The witness is bound to the coordinates and actor it attests to, so
neither a direct construction nor a dataclasses.replace off a
legitimate decision can re-target one. Its only callers are tests until
the flow guard lands; that exception and the booked move of the type to
scufris_core are recorded in the task's DECISION.md.
```
