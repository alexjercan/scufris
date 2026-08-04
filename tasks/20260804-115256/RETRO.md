# Retro: Record the chat conversation and event tables with typed actors

- TASK: 20260804-115256
- BRANCH: feature/chat-conversation-events
- REVIEW ROUNDS: 3

## What went well

- DECISION.md settled five forks before the first line was written, and not one
  was re-opened during implementation or review. The three rounds argued about
  invariant ENFORCEMENT, never about what to build.
- The store as functions over an open `Connection` held up under review: nothing
  proposed a class, because there is no second unit of work for one to hold.
- Every guard was sabotaged before it was trusted, in all three rounds, by the
  implementing side and again by the out-of-context reviewer. That is what made
  round 3's "wider than asked" fix checkable rather than a claim.

## What went wrong

Eleven findings, and the same defect three times: an invariant enforced one and
a half times.

1. R1.2 - `actor_kind` was CHECK-constrained; `actor_agent_id` was not.
2. R1.3 - the kind list could not drift model-to-enum, but could
   enum-to-shipped-revision, because `compare_metadata` does not diff CHECKs.
3. R2.1 - the R1.2 fix wrote the rule as NULLABILITY while `__post_init__`
   states it as TRUTHINESS, so `('agent','')` walked through the difference.

The decision that failed was treating "the CHECK constraint" as one thing to
add. It seemed sound because the constraint WAS added, in the model and in the
revision, and the whole suite went green - the tree gave no signal that a
predicate had been half-translated. The root cause is that the rule lives in two
languages, Python truthiness and SQL nullability, and the translation was done
by eye rather than by enumerating the column's falsy values.

Two plan gaps of one shape, both about the root distribution's edge to the new
member: Step 8 forbade the `DECLARED_GRAPH` entry that Step 7's `env.py` import
creates, and no Step named `[project.dependencies]` / `[tool.uv.sources]` at all
(R1.1). The plan reasoned about the IMPORT and never about the DISTRIBUTION that
contains it.

## What to improve next time

- Breadth is not the lesson here. ~1600 lines is what a workspace member carve
  costs - distribution, two tables, a migration, an example, four doc surfaces -
  and no part of it lands independently: the member gates
  (`DECLARED_GRAPH`, `EXAMPLES_BY_MEMBER`) go red the moment the directory
  exists and stay red until the example and the graph entry land. The plan named
  that red and sequenced it, which is the correct handling.
- Churn is the lesson, and the plan-time question that would have caught all
  three is `plan/decision.md`'s cold-reader rationale test applied to the
  INVARIANTS rather than to the design: for each rule the code states, where
  else is it stated, in what language, and what does a reader who only has the
  schema conclude? Steps 3 and 4 listed columns and constraints as artifacts to
  create. An invariant table - rule, code site, schema site, test that proves
  they agree - would have made R1.2 and R2.1 the same missing row.
- When an invariant lives in both a dataclass and a CHECK, derive the SQL from
  the Python predicate's exact semantics, not from what the column's nullability
  suggests. `not x` and `x IS NOT NULL` differ on every falsy value the column
  can hold.
- Anything hand-written into a revision - CHECK text, a trigger, a partial index
  - is outside `test_schema_has_no_pending_autogenerate_diff`'s guarantee and
  needs its own assertion against a migrated database.
- New workspace member: the checklist is wider than the source tree. Beyond
  `pyproject.toml` and `uv lock`, a member the ROOT imports also needs
  `[project.dependencies]`, `[tool.uv.sources]`, `known-first-party`, the
  `DECLARED_GRAPH` edge, `EXAMPLES_BY_MEMBER`, and four doc surfaces. Half of
  those went in as review findings.

## Context

No context pressure was recorded: no checkpoint, no handoff, no compaction
warning across the three rounds. Round 3's review used the standard
out-of-context reviewer, which is procedure rather than pressure. Nothing to
split, delegate or defer on this evidence.

## Action items

- Two NITs remain open in REVIEW.md round 3 and are deliberately not fixed here:
  R3.1 (the facade docstring names one CHECK where there are now two) and R3.2
  (`__post_init__`'s docstring overclaims for an untyped `kind`). Neither
  affects behaviour reachable through `parse` or `_record`, and mypy rejects
  R3.2's input repo-wide. The lane that next touches `packages/chat` should
  close them; R3.1 is one sentence.
- The intermittent `tests/test_app.py::test_agent_run_reaches_done_and_persists_
  session` flake recorded in the close-out is not this task's and did not
  reproduce. It stays a recorded observation, not a task, until it is seen twice.

## Knowledge

Three lessons written to `/home/alex/personal/agent-knowledge`, all new, with
`knowledge check` clean:

- `pattern/an-invariant-restated-in-another-language-needs-an-exact-translation`
  - the truthiness-versus-nullability root cause of R1.2 and R2.1.
- `testing/hand-written-schema-text-escapes-autogenerate` - R1.3: what a
  "no pending diff" test does and does not guarantee.
- `changes/an-import-creates-a-distribution-dependency` - R1.1: a
  registration-only import is still a packaging edge, and a dev checkout hides
  the missing declaration.

No failed writes.

## Landing message

```
feat: record the chat conversation and event tables with typed actors

Adds packages/chat, the sixth workspace member, with `conversation` and
`event` declared against scufris_core.Base, an Actor value over the four
ratified kinds, and a store of four functions over an open Connection - the
signature is what makes a state change and its event commit together.

event_seq is per-conversation, assigned inside the caller's transaction and
backed by a unique constraint; the actor rule is held in the schema by two
CHECKs stated as truthiness, so a hand-written INSERT meets the same rule
Actor does. Revision 18c9104709b8 ships both tables. Nothing reads a
conversation yet.
```
