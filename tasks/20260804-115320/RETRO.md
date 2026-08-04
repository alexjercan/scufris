# Retro: Assemble provider context from the semantic conversation

- TASK: 20260804-115320
- BRANCH: feature/chat-provider-session
- REVIEW ROUNDS: 3

## What went well

The DECISION-first Step earned its place twice over. Sections 1 and 3 changed
the schema and deleted a whole code path - two-column primary key rather than
the triple, and no eager re-seed at all - so writing them after the code would
have meant rewriting it. Worth repeating on any task whose plan says a record
decides the shape.

Bounding the assembly in SQL rather than slicing a full read was specified in
the plan rather than found in review, and the generalization it names
(`format_fork_seed` slices after loading everything) is the kind of pointer a
plan can carry cheaply and a reviewer cannot.

## What went wrong

The same defect was found twice, by two different rounds, and the second finding
is the one worth keeping.

R1.1 raised an unvalidated `agent_id` forging an `operator:` line in the
assembled prompt. The fix refused every C0 control and DEL, and its own
close-out prose named the correct general shape: re-validate a value AT the
domain crossing, because the format that makes the body safe is exactly what
hides the attribution. R2.2 then showed the fix still let the forgery through
one character over: `str.splitlines` also breaks on U+0085, U+2028 and U+2029.

The failed decision was choosing the alphabet from the value's OWN domain - "an
id should not contain control characters" - when the rule that mattered belonged
to the consumer, the function that decides where a line ends. It seemed sound
because C0-plus-DEL is the standard answer to "what may an identifier not
contain", and it is the right answer to that question. It is the wrong question.
Naming the crossing was not enough; the rule had to be derived from the code on
the far side of it.

R2.2's own suggested derivation was also wrong - `("x" + c).splitlines()` is
length one for every code point, since a trailing terminator ends no second line
- which is a second instance of the same thing: a rule quoted from a review is
still a rule to re-derive. The shipped test uses `f"x{c}y"` and enumerates every
code point, so the list is now held to the splitter rather than to anyone's
memory of it.

R2.1 was the other real one: the close-out recorded "all 9 checks pass" for a
`nix flake check` that declares six, and a pytest count that a later fix had
already invalidated. Both were written from memory at close-out time rather than
from the run. The evidence now names the six checks instead of counting them,
which is a number that cannot go stale without being wrong about something
checkable.

## What to improve next time

**Breadth.** The diff is large but not split-shaped: four Steps out of eleven
are the same table's model, migration, store surface and schema test, and none
of them lands without the others. The one genuine boundary - the retrospective
`event` CHECK tightening in revision `7f21c0d4ae90` - arrived from review, not
from the plan, and by then it belonged in the same table rebuild. No missed
split.

**Churn.** All three rounds were spent on one class of question, and it is one a
plan-time challenge would have surfaced: the Step that specifies the attributed
render says a hostile BODY becomes a quotation, and never asks what makes the
ATTRIBUTION itself trustworthy. The cold-reader test in `plan/decision.md` would
have caught it - a cold reader asked "why can I trust this prefix?" has no answer
in the plan. Worth adding to the plan-time checklist for any format whose safety
comes from structure: name every field the structure interpolates, and say what
validates each.

**Context.** No pressure observed. No checkpoint, no compaction warning, no
delegation beyond the standard out-of-context review rounds. The round-2 fix
pass touched nine files and stayed inside one focused pass.

One process signal from round 3 that belongs to a later lane rather than here:
`downgrade()` has no automated coverage anywhere in this repository - the suite
only ever migrates forward - so this revision's round trip was exercised by
hand. That is the repository's convention, not this diff's omission, and a lane
should decide on it deliberately.

The `store.py` split signal stands from all three rounds: 582 lines against the
600 cap, owning four tables plus the prompt-assembly constants. The next chat
lane has to plan the split; it has 18 lines of headroom.

## Action items

- Central knowledge submission: validate a value against the CONSUMER's
  alphabet, not its own domain's, at every crossing - and pin it with a test
  that asks the consumer rather than restating the list.
- Next chat lane plans the `store.py` split before adding to it.
- A lane decides whether Alembic `downgrade()` paths get automated coverage.

## Landing message

```
feat(chat): assemble provider context from the semantic conversation

Add the `provider_session` cache and windowed context assembly, so a backend
switch, a `/new` or a restart re-seeds from the conversation rather than
losing it. The binding is keyed by `(conversation_id, backend)` with
`policy_version` as a matched column, so a policy downgrade cannot resurrect a
superseded session; re-seeding is lazy, driven by the cache miss itself.

Assembly is bounded in SQL rather than by slicing a full read, and every line
names its author under a preamble saying only the operator's lines are
instructions - an agent report reaches the provider as a quotation. The `event`
CHECKs are tightened to match: no empty body, and no agent id containing
anything `str.splitlines` treats as ending a line, since either would reach the
seed prompt as a missing line or a forged attribution.

Summarization and a character-or-token bound are recorded deferrals with the
triggers that reopen them.
```
