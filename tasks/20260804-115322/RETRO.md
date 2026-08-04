# Retro: Prove Lane 1 with the conversation demo and the chat explainer

- TASK: 20260804-115322
- BRANCH: feature/chat-lane1-demo
- REVIEW ROUNDS: 3

## What went well

Cutting the deliverable out as its own record worked exactly as the Notes
predicted it would. Nine of the eleven review findings across three rounds are
about the demo and its gates, and every one of them would have been the part
folded into a build task and dropped.

Deriving assertions from the demo's own source rather than re-typing its
constants - `_causation` walking the `append_event` calls, `TREE_GUIDES` and
`depth` taken off `demo` - is what made the later rounds cheap. A finding could
be stated as "this derived value feeds the wrong predicate" rather than "these
two copies disagree".

Sabotage as the unit of evidence. Every round's Response names a mutation, the
message it produced, and the fact that it was reverted. That is what let round
3 confirm three fixes in one pass, and what let round 2 discriminate the old
predicate from the new one when no red run separated them.

## What went wrong

The same defect shape recurred in all three rounds: an assertion whose name and
docstring claim more structure than the assertion carries.

- Round 1, four findings: `label in stdout` for attribution, event 1 hardcoded
  as every event's parent, re-typed constants.
- Round 2: `depth(answer) > depth(asked)` pins "deeper than SOME ancestor", not
  "under the event it answers". Round 1's R1.3 had fixed how the causation
  edges are DERIVED and left the predicate consuming them untouched - the fix
  read as complete because the derivation it described was.
- Round 3: the gate COUNTS two renderings and never compares them, and
  `parent_line` resolves its target by value on a list with duplicates.

The root cause is a split in where the claim lives. The demo asserts a property
precisely (`after_rendered != rendered`, `tree_problems`); the gate then
asserts a weaker proxy for the same property (a count of two, a depth
inequality) while its message repeats the demo's stronger wording. The gate
inherits the demo's prose and none of its precision.

The failed decision, and why it seemed sound: the Step said "put at least one
assertion behind every claim the new output makes", and a count-of-two IS an
assertion behind "rendered twice". Read at plan time that reads complete. What
it does not say is "and pin each assertion to the structure that carries the
claim, not to a proxy that correlates with it".

## What to improve next time

For a task whose deliverable IS the proof, add a plan-time step that names the
falsification for each assertion: not "assert X" but "assert X, which goes red
when Y". An assertion whose Y cannot be written down is a proxy. Both round-2
findings and both round-3 findings would have been visible at plan time under
that rule, because none of them has a writable Y - "the tree is redrawn" does
not fail a depth inequality, and "the second rendering drifted" does not fail a
count.

Where a demo and its gate check the same property, the rule belongs in ONE
function that both call, as `parent_line` now is. Round 2 fixed two findings
with one helper; R3.1 and R3.2 are the same pair left over on a different
property.

**Breadth.** The diff is large (1756 insertions) and correctly so: 1056 lines
are `chat.html`, a document, and 262 are the demo. No independently landable
split was missed - the explainer and the demo are the two halves of one
deliverable, and splitting them would have landed a lane whose proof and whose
explanation arrived separately, which is the failure the record was cut to
prevent.

**Churn.** Three rounds, eleven findings, no rework of the SHIPPED behaviour -
`render_transcript` is unchanged in substance from round 1 to round 3, and only
the gates and one docstring moved. The plan-time question that would have
prevented it is `plan`'s from-scratch challenge applied to the TESTS rather
than the code: the Step listed what to assert and never asked what each
assertion would fail on. The plan is the subject here; the worker wrote what
the Step asked for.

**Context.** No pressure observed. No checkpoint, no compaction warning, no
handoff. Each review round ran in a fresh out-of-context reviewer, which kept
the recording pass small enough that all three rounds fit one working context.

## Action items

- R3.1 and R3.2 ride into the retro rather than a fourth round, by the round-3
  verdict's own reasoning: R3.1's property is guarded by the demo's exit code
  and R3.2 is latent while the two rendered blocks stay byte-identical. They
  become live together, so whoever fixes one fixes both.
- `tasks/20260804-173304` is seeded for the `test_app.py` order-dependent
  flake this branch observed twice. Not this branch's defect; its record now
  names both tests and asks for a recorded `--randomly-seed`.

## Landing message

```
feat(chat): prove the conversation lane with a runnable demo and an explainer

Grow `examples/chat_conversation.py` into Lane 1's proof: a real database, a
conversation written in one transaction, an `authorize` decision minted from
the operator's message and refused for the agent's report, the transcript
drawn as an attributed causation tree, two channels delivering, and a backend
switch that changes the provider session and nothing else. The demo renders
into a recording console and asserts on the exact string it prints, so its
exit code is the verdict.

Two gates in `tests/test_examples.py` read that output back: every function
`scufris_chat` exports is called by the demo, and the printed transcript
carries every event, its author, and the edge to the event it answers - each
assertion derived from the demo's own source rather than re-typed beside it.

Add `tasks/20260801-154211/chat.html`, the lane's explainer: the event model,
the four owned records and who writes each, the settled per-turn granularity,
and the retention non-decision. Linked from the package README.
```
