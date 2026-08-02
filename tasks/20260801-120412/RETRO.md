# Retro: Cut the project store over to the database

- TASK: 20260801-120412
- BRANCH: fix/project-store-database
- REVIEW ROUNDS: 2

## What went well

- The headline proof was written first and against the real API, so the epic's
  claim was measured through the routes rather than asserted about the store.
- The 20260729-102146 repro harness was rewired onto the replacement instead of
  deleted. The before/after numbers now come from one rig (200 recovered against
  103), and the three stores that have not moved still fail in it.
- Deleting `migrate_state_dir` rather than keeping it alongside
  `open_state_database` left one startup entry point for the next cutover to
  copy, with no wrong one to pick.

## What went wrong

- Three `async def` agent routes kept calling the store directly, so a
  `BEGIN IMMEDIATE` transaction ran on the event loop thread. Review measured
  the loop stalled 3.04s behind a held write lock against a 0.01s heartbeat;
  past `busy_timeout` it becomes a 500. The plan's sweep Step named
  `ProjectStore(...)` CONSTRUCTION sites and every one of them was checked - the
  fault is that the question was "who builds this", when moving a dict read onto
  a lock-taking transaction changes the cost model for everyone who CALLS it,
  and the offending call was one indirection away in `_require_agent_project`.
- `update` began validating its fields before resolving the record, turning a
  404 into a 422 for an unknown id. The Step said "keep the observable
  behavior"; the reordering came out of hoisting the `is_dir` stat outside the
  transaction, which was a real concern applied without checking what it moved
  past.
- Two operator-visible changes rode in without their own records: the
  `_TASK_LINE_RE` fix (a pre-existing red test, defensible to fix here since it
  gated this task's own proof, but it shipped with no changelog entry until
  review asked) and `create_app` now refusing to start on a damaged
  `projects.json` - documented in README and the changelog, untested at the
  boundary where it changed, because the test that pinned the old tolerant
  behavior was deleted rather than replaced.

## What to improve next time

- Breadth: 22 files, ~1075 insertions. The cutover itself cannot be split - the
  store, its callers and the startup wiring have to move together - but two
  pieces inside it could have landed separately: the `scufris/mcp_stores.py`
  extraction, which the 600-line file cap forced mid-task, and the ~40
  mechanical test-fixture call-site edits across eight files. Both are
  independently landable and both will be touched again by the next two
  cutovers.
- Churn: one MAJOR, and the plan-time question that would have prevented it is
  not the from-scratch challenge - the design was right - but a cost-model
  question the plan never asks. A Step that says "grep every construction site"
  should say "grep every CALL site and name the thread each runs on" whenever
  the thing being moved changes from a memory read to a lock acquisition.
- Context: no measured pressure. No checkpoint or compaction warning is
  recorded. Review round 1 ran in a fresh session that started at REVIEWING,
  which is the intended out-of-context handoff and cost nothing extra.

## Action items

- The remaining cutovers, 20260801-100409 and 20260801-100413, inherit the
  event-loop hazard verbatim: their stores have callers inside `async def`
  routes too. Each needs a call-site-by-thread sweep in its Steps and a test
  like `test_project_lookup_never_runs_on_the_event_loop`.
- Submitted to central knowledge: the call-sites-by-thread sweep for a store
  moving onto a locking transaction.
