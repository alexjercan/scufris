# Retro: Migrate auth, host, schedule, and digest state with a legacy JSON import path

- TASK: 20260801-100413
- BRANCH: fix/db-post-host-state
- REVIEW ROUNDS: 2

## What went well

The plan named the risk correctly. Its Notes called the `asyncio.to_thread`
offloads "the largest and riskiest part of this task, not the schema", and that
held: the offloads touched four modules and every one landed, with the engine's
event-loop guard turning any miss into a loud error rather than a stalled lock.
Neither review finding was in that area.

Two claims the branch made about concurrency were re-derived independently at
review rather than taken from the branch's own tests, and both survived: 40
threads racing one proposal produced exactly one decision and 39
`AlreadyDecided`, and 30 concurrent `put`s produced no `seq` collision.

## What went wrong

Both findings share one root: a record asserted a property that nothing
executed.

`import_legacy_state`'s docstring claimed a refusal "degrades correctly" and
that the retry after the repair reads only the damaged file. The first half was
true and the second was never walked by a test. It was false, and expensively
so - a refused `sessions.json` let `agents.json` import and gate, `load_agents`
migrated the pre-registry `session_id` into `agent_session` itself, and the
repaired file then hit `UNIQUE constraint failed` that no retry could clear,
because the gate row means the conflicting write is never replayed. The
documented recovery path was a permanent unbootable startup.

Why it seemed sound: the existing test
`test_a_damaged_source_does_not_hold_back_the_other_sources` parameterises over
a front and a back source and asserts every OTHER source still gates. That reads
like full coverage of the refusal policy, and it is - of the refusal. It stops
at the point the operator's work begins. The prose then described the repair as
though the test had covered it.

`test_post_host_state_uses_declared_persistence_boundary` listed its six stores
by hand, so the DoD line "every app-owned store shares the declared boundary"
could not fail for a store the list did not know. Replacing the list with a walk
over `app.state` immediately falsified it: `ConfigChangeStore` is still an
in-memory `OrderedDict`, unnoticed by the task, its plan, and the epic.

## What to improve next time

Breadth: 3030 insertions across 37 files is large but not a missed split. Four
store cutovers plus a shared import path is one boundary closing, and the plan's
own Notes explain why the migration docs and the whole-directory import belong
to this task specifically ("the first point at which every store is on the
core"). The one real split was found at review and taken as 20260803-002141,
correctly - migrating a fifth store was outside these Steps.

Churn: `plan`'s cold-reader rationale test would have caught R1.1. The "degrades
correctly" sentence was inherited prose, carried from master's
`import_agent_state` into the new entry point and then widened from two per-half
loops to one whole-directory loop. A cold reader asked "which test walks this?"
gets no answer. Inherited prose is the dangerous kind: it reads as settled
because it has been in the tree, and moving it into a wider scope is exactly
when its claim needs re-earning.

Context: no measured pressure. No compaction warning, no checkpoint, no
delegation; review and both fixes ran in one pass.

## Action items

- 20260803-002141 - move `ConfigChangeStore` onto the database, and delete the
  `config_changes` exclusion the boundary test now carries. Filed under the same
  epic, discovered by this task's review.
- For the epic's remaining store cutovers: a docstring that claims a RECOVERY
  path owes a test that walks the recovery, not only the failure. A DoD line
  that quantifies over a set ("every app-owned store") owes a test that derives
  the set rather than restating it.
- Watch for inherited prose when code MOVES. Both this branch's package split
  and its entry-point consolidation carried docstrings into wider scopes, and
  the claim that was true of the old scope was the one that broke.
