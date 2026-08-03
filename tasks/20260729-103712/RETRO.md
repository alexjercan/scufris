# Retro: Extract the remaining routers and reduce create_app to assembly

- TASK: 20260729-103712
- BRANCH: refactor/extract-remaining-routers
- REVIEW ROUNDS: 2

## What went well

- One atomic green commit per Step held for all ten Steps: 19 commits, 36
  files, 5285 insertions against 2201 deletions, and every commit left the
  suite, the linter, the type checker and the size check green. That is what
  made the two mid-task checkpoints hand off on disk with nothing uncommitted.
- `tests/test_route_contract.py` was written on the BASE branch as a
  characterization baseline before any code moved - route table, middleware
  order, `app.state` keys, the five `create_app` override points. It is green
  on base by construction, so it was never a DoD proof, but it is what made a
  2000-line move reviewable as a move rather than as a rewrite.
- Proofs were checked red on base at plan time, including the one-way
  interaction between the `ALLOWLIST` ratchet and the split: dropping the entry
  without splitting fails on the cap, splitting without dropping the entry fails
  as stale, so the pair could only go green together. That forced the last Step
  into one commit rather than being discovered as a broken build.
- The `_forbidden` trap pattern - booby-trap `Settings.__init__`,
  `AgentStore.__init__`, `ProjectStore.__init__`, `Database.__init__` and
  `state_database`, then drive the router on a bare `FastAPI()` - turned "the
  router binds its dependencies" from a claim into a falsifiable test, and its
  absence on one router is exactly what review round 1 was able to find.

## What went wrong

- The Steps budgeted ONE test file for six routers
  (`tests/test_orchestrator_routers.py`). It hit its 900-line cap three times,
  each discovered mid-flight: `tests/test_chat_router.py`, then
  `tests/test_legacy_agent_router.py` (both recorded as plan corrections), then
  `tests/test_agent_run_router.py` - and the third was forced by review finding
  R1.1, not by the worker. The failure is the shared file, not the cap: when
  the file was full, the agent-run slice landed its rig WITHOUT its `_forbidden`
  pass, so DoD 2's claim silently covered four routers of five. It looked sound
  at plan time because `tests/test_domain_routers.py` had carried three routers
  comfortably - the plan measured that file's size but not its slope.
- The plan's line projection for `app.py` (~575) was built from measured
  segments of the real file and still landed 93 lines optimistic (668 after the
  eight route/service Steps). The named fallback did not close the gap alone,
  so two further blocks left the factory beyond what the plan authorized. The
  projection counted only lines LEAVING, never the call site each extraction
  leaves behind: an `include_router`, a deps construction, and the comment.
- A fake `launch` returning an `EventBus` nobody closes hung the entire suite -
  no failure, no timeout, no output. Diagnosed only by `kill -SIGABRT` on the
  stuck pytest and reading the faulthandler dump. Cost: a debugging pass that
  produced no diff.
- One bind site was missed by grep and found by the suite: a logger NAME inside
  `caplog.at_level(..., logger="scufris.app")`. The Steps enumerated import
  bind sites carefully; string-literal module names were not in that sweep.

## What to improve next time

- Breadth. The diff is large and that is inherent, not a missed split: it is one
  file going from 2621 lines to 586, the epic's Lane D had already taken the
  domain routers out ahead of it, and the ALLOWLIST ratchet forces the final
  Step and the entry removal into a single commit. No further independently
  landable split was available.
- Churn. Both review rounds trace to one plan-time gap, and the cold-reader
  rationale test in `plan/decision.md` is what would have caught it: the Steps
  said WHERE to assert (one file) instead of WHAT each router owes. A move
  extracting N routers should budget N test files and state the evidence as N
  per-router claims, so a full file re-homes a slice instead of dropping its
  trap.
- Estimation. Budget roughly 10 lines of residual call site per extracted module
  when projecting a post-split file size, on top of the lines that leave.
- Sweeps. When moving code between modules, grep for `scufris.<module>` STRING
  LITERALS as well as for imports. `caplog` logger names, `monkeypatch.setattr`
  targets and patch paths all bind by string and break silently.
- Context. Two checkpoints were taken, both at a Step boundary with a clean
  tree, and the second deliberately BEFORE the largest remaining Step rather
  than part-way through it. Repeat that. Both review rounds went to an
  out-of-context reviewer: round 1 found the DoD 2 hole the implementing context
  had ticked, and round 2 found the residual `/fork` gap inside round 1's own
  fix. On a task this size the delegation is carrying real weight, not
  ceremony.

## Action items

- Three review findings are open and unfixed, all MINOR or NIT, none blocking
  the APPROVE: R2.1 (the agent-run trap docstring claims `/events` is the only
  excluded route, but `/fork` is excluded too, so coverage is 14 of 16),
  R2.2 (`scufris/README.md:85` still cites `app._build_telegram_approval_ops`,
  which this branch moved to `telegram/wiring.py::build_approval_ops`), and
  R2.3 (an unused module logger in `scufris/host_approval_bridge.py`). Seeded
  as task 20260803-102351 rather than reopening this branch.
- `tests/test_app.py::_wait_state` polls a background run 200 times at 10ms and
  then RETURNS the last state instead of failing, so a lapsed deadline surfaces
  as a confusing assertion about a session id. Load-dependent flake, untouched
  by this branch. Already recorded as task 20260803-100411.

## Knowledge

Written to `/home/alex/personal/agent-knowledge` (project=scufris), all four
accepted and `knowledge check` clean:

- `planning/a-step-inherits-its-files-constraints` - occurrence added. The
  shared test file hitting its cap three times is the same lesson seen from the
  planning side; the note records that N Steps writing to one file need N files
  budgeted up front.
- `testing/patch-where-the-name-is-bound` - occurrence added. The traps had to
  patch `Settings.__init__` rather than `scufris.config.Settings`, because the
  routers do `from ..config import Settings` and the name is bound into the
  importing module at import time.
- `changes/moving-code-renames-its-string-bound-identities` - new. Identity
  derived from `__name__` moves with the code, and every reference to it is a
  string no import sweep or type checker sees.
- `planning/budget-the-call-sites-a-split-leaves-behind` - new. A projection
  built only from the segments that move is systematically optimistic, and the
  error grows with the number of extractions.
