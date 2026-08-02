# Retro: Move the configuration-change registry onto the database

- TASK: 20260803-002141
- BRANCH: refactor/config-change-registry-db
- REVIEW ROUNDS: 1

## What went well

The plan named the one thing that was not `HostActionStore` again, and it named
it from reading the code rather than from the diff: `ConfigChangeBuilder.stream`
mutates the stored object, so the existing HTTP tests observe transitions for
free and would observe nothing against a row. The Notes went further and named
the regression net - `_settle` polls the GET route, so a missed write-back hangs
ten seconds and fails rather than passing quietly. Nothing in that area was a
review finding.

The second discovered consequence was the more valuable one, because it was not
a mechanical consequence of the cutover at all. Durability removes the implicit
clear a restart used to give, and a change left `building` by a crash would have
answered `building_for` forever - a 409 on every later build of that repository
that the documented escape cannot clear, because cancelling needs a live
supervisor. `abandon_builds()` and the second new test exist because planning
asked what durability TAKES AWAY, not only what it adds.

One APPROVE. The three findings are all MINOR or NIT and none touches the
migration.

## What went wrong

The plan wrote `Save = Callable[[ConfigChange], None]` and it had to be awaited.
`stream` is an async generator the supervisor drives on the event loop, and
`Database.transaction()` refuses a thread with a running loop, so the planned
signature would have raised on the first transition of every build.

Why it seemed sound: the offload rule is written, and enforced, as a rule about
`async def` ROUTES. Step 5 says so in those words and lists the four route call
sites to grep for. `save` is the first writer in this codebase that is neither a
route nor a synchronous startup path - it is a callback into a supervisor task -
so the rule as stated did not reach it while the hazard did. The engine guard
caught it immediately, which is the argument for having made that an exception
rather than a comment.

The review's R1.1 is the sharper failure, and it is a recurrence. Both restart
proofs hold the first `TestClient` open across the "restart", and `create_app`
takes its handle from the process-wide `_HANDLES` memo
(`scufris/db/__init__.py:45`), so the restarted app is handed the same
`Database` and the same pool. Re-derived here rather than taken from the review.
They are honest tests of "no longer a per-app dict" - both are red on the base -
but their docstrings claim the change and its proposal "outlive the process",
and nothing walks that.
`test_the_digest_store_survives_a_restart_and_stays_bounded` is the repo's
pattern for it and even says why in one line: "Reopened rather than shared".

That is the same root the previous cutover's retro (20260801-100413) wrote down
as its action item: a record asserting a property that nothing executes. It
recurred one task later, in the DoD's central proof, with the pattern already in
the tree.

Two tests broke on facts they never mentioned. `test_a_damaged_database_...`
asserted `sqlite3.DatabaseError`, which held only while the corrupted pages were
reached first by `current_revision`'s raw read rather than by `open_database`'s
pragma dial through SQLAlchemy; one more table's worth of schema pages moved
which read trips. `test_the_backup_is_taken_on_the_real_migration_path` and
`test_declared_tables_are_the_only_ones` both name the head revision's tables.

## What to improve next time

Breadth: 558 insertions across 14 files, and not a missed split. One store, one
revision, one write-back seam and its two docs, which is the unit the plan
scoped. The `save` callback could not have landed separately from the store it
writes to.

Churn: the round produced no rework, so the question is which plan-time question
would have caught R1.1 before it was written. `plan/decision.md`'s cold-reader
rationale test, asked of the test DOCSTRING rather than of the prose: "outlive
the process" - which line opens a second engine? There is no answer, and the
sentence next to it in `test_host_digest.py` is the answer it should have
borrowed. A restart test in this repo owes a reopen, the way a recovery
docstring owes a walk. Both are the same rule about claims and executors, and
this is its second instance.

Context: no measured pressure. No compaction warning, no checkpoint, no
delegation; plan, work and review each ran in one pass.

## Action items

- 20260803-014401 - reopen the database in both restart proofs, cover the
  durable `_reap` bound, and settle R1.3. Carries R1.1-R1.3 verbatim.
- A restart test that shares the process handle proves the store left the
  process's dict, not that the row is on disk. Here that means closing the first
  client or calling `close_state_database` before rebuilding the app - `_HANDLES`
  will otherwise hand back the same pool and the test passes for the wrong
  reason. Generalises the previous cutover's "a docstring that claims a recovery
  path owes a test that walks it"; the pair is now two instances of one rule and
  belongs in central knowledge, not in a third close-out.
- The event-loop offload rule is written for `async def` routes and the hazard is
  wider: any writer reached from the loop, including a callback into a supervisor
  task. Filed separately as 20260803-014210, where the same gap makes
  `create_app` unrunnable from a running loop in three examples.
- A migration task should budget for tests that encode "the current head adds
  table X" or "the corrupt page is reached by the second read". Three paid that
  toll here; expect it each time a table is added.
- `_row` / `_values` / `_change` / `_reap` are now a second near-verbatim copy of
  `host_actions.py:363-403`. Deliberate at two instances. A third store is the
  trigger for a shared row-store helper, not this one.
