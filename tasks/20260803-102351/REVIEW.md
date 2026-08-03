# Review: Close the round-2 findings from the create_app assembly extraction

- TASK: 20260803-102351
- BRANCH: refactor/close-round2-findings

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tasks/20260803-102351/TASK.md:105 - the close-out's
  Evidence paragraph says "All four `cmd:` proofs ran red on the base and green
  after" and then enumerates the fourth as `python -m pytest && ruff check . &&
  mypy .`. That proof is green on the base: run at `ece331a` it exits 0 with
  `1108 passed, 1 skipped`, ruff clean, mypy clean over 229 files. A
  regression-guard proof cannot be red on base. Reword to say the three
  targeted proofs ran red on the base and green after, and that the suite proof
  was green on both.
  - Response: fixed in this round's commit - the Evidence paragraph now says
    the three targeted proofs ran red on the base and green after, and names
    the suite proof separately as a regression guard green on both base and
    branch.

- [x] R1.2 (MINOR) tests/test_agent_run_router.py:13 - the module docstring
  still reads "The rig's fakes are imported from `test_orchestrator_routers`;
  only the diagnostics fake is redefined", which this diff makes false -
  `ForkingRunService(FakeRunService)` is a second redefinition, and DECISION.md
  cites that very sentence as its precedent. Amend it to name both, e.g. "the
  diagnostics fake and the run-service fake are redefined, because the run
  surface asks diagnostics for `health`, `tools` and `mcp`, and asks the run
  service for `fork_seed`."
  - Response: fixed in this round's commit - the module docstring now names
    both redefinitions, in the reviewer's words.

### Verified

- Full suite on the branch: `1108 passed, 1 skipped`, exit 0; `ruff check .`
  clean; `mypy .` clean over 229 files. Matches the close-out's numbers.
- All four `cmd:` proofs run from the worktree: exit 0 each. No `manual:`
  proofs in the Definition of Done, so there are no pending user checks.
- Re-derived independently for R1.1: `python -m pytest` on `master` at
  `ece331a` exits 0 with `1108 passed, 1 skipped`; `ruff check .` and `mypy .`
  both exit 0. Proof 4 is green on the base.
- Re-derived independently for R1.2: `tests/test_agent_run_router.py:13-15`
  still carries the "only the diagnostics fake is redefined" sentence.
- R2.2: `_build_telegram_approval_ops` appears nowhere under `scufris/`,
  `tests/` or `web/`; `scufris/README.md:85` names
  `telegram/wiring.py::build_approval_ops`, which exists at
  `scufris/telegram/wiring.py:51` and is in that module's `__all__`. The doc
  sweep for the old symbol hits only `tasks/`, which is exempt.
- R2.3: `scufris/host_approval_bridge.py` carries no `logging` token, and no
  other module imported its `logger`.
- R2.1b delivers its Step's literal text: the added assertion is the one the
  Step specifies, placed after the `/chat` assertion. `api/agent_runs.py` has
  16 routes and the trap now drives 15, so the docstring's "``/events`` is the
  one route left out" is accurate as written and correctly needed no rewording.
- The new assertion cannot pass vacuously: `/fork` returns 200 only past its
  404, 409 and 422 arms (`scufris/api/agent_runs.py:416-448`), so it pins that
  the route body reached `fork_seed`. Deleting `ForkingRunService` breaks the
  test, because the shared `FakeRunService` has no such attribute.
- DECISION.md's premise holds: `tests/test_orchestrator_routers.py` is 897
  lines against the 900-line cap in `scripts/check_file_size.py`, and the diff
  leaves that file untouched.

Process signal: DECISION.md flags splitting `test_orchestrator_routers.py`
along its three rigs as wanting its own task, and the 897/900 headroom is now
load-bearing - the next addition to any of those rigs hits the same wall. No
task is filed for it yet.

Process signal: the plan named a file and line for R2.1a without checking that
file against the repo's own line cap, which cost a mid-work correction and a
DECISION.md. `wc -l` against `scripts/check_file_size.py` is a planning step.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (NIT) tasks/20260803-102351/DECISION.md:45 - the Consequences
  section argues its precedent from a present-tense quote of the module
  docstring, "only the diagnostics fake is redefined, because the run surface
  asks it for `health`, `tools` and `mcp` as well as the three the shared one
  answers". Round 1's R1.2 fix rewrote that sentence
  (`tests/test_agent_run_router.py:13-17`), so the quote no longer exists in
  the file it cites. Either requote the amended sentence or drop the quotation
  and cite the `FullDiagnostics` redefinition directly.
  - Response:

### Verified

- R1.1 confirmed fixed: `tasks/20260803-102351/TASK.md:102-108` now separates
  the three targeted proofs (red on base, green after) from the suite proof,
  which it names a regression guard green on both base and branch. Matches the
  observed reality recorded in round 1.
- R1.2 confirmed fixed: `tests/test_agent_run_router.py:13-17` names both
  redefinitions in the reviewer's wording. `ForkingRunService(FakeRunService)`
  and `FullDiagnostics` are the two, and the `fork_seed` claim is true of the
  rig.
- Fix commit `2a9edda` touches exactly three files (TASK.md, REVIEW.md,
  `tests/test_agent_run_router.py`); the test change is docstring-only, so it
  carries no behavioural regression surface.
- Full suite on the branch: `1108 passed, 1 skipped`, exit 0; `ruff check .`,
  `ruff format --check .` and `mypy .` all exit 0, mypy over 229 files. The
  1108 count re-derived in the recording pass with `-p no:cacheprovider`, which
  restores the summary line the repo's config suppresses.
- All four `cmd:` proofs run from the worktree: exit 0 each. No `manual:`
  proofs in the Definition of Done, so there are no pending user checks.
- R2.1 re-derived independently in the recording pass: the quoted sentence
  appears at `DECISION.md:45` and matches nothing under `tests/` or `scufris/`.
- No other file under `scufris/`, `tests/` or `web/` quotes or depends on the
  amended docstring sentence.

Process signal: both round-1 findings were record-accuracy defects, not code
defects - a close-out overclaiming which proofs were red on base, and a
docstring the diff falsified. Neither is reachable by the suite. The proofs a
task writes about itself want the same red-on-base discipline the task's own
`cmd:` criteria get.
