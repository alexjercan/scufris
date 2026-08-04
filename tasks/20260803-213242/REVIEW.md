# Review: Carve the code into a uv workspace of per-service packages

- TASK: 20260803-213242
- BRANCH: master

No feature branch: this is an EPIC, and its six children each landed on `master`
through their own branches. The subject of the review is therefore the
cumulative tree, judged against the epic's own Done Means.

## Round 1

- REVIEWER: out-of-context `general-purpose` subagent (id `a480234221fe27c62`),
  given only the task ID, the repository path, the review dimensions and the
  finding format.
- VERDICT: REQUEST_CHANGES

Primary re-derived, independently of the reviewer: the `core` module listing
against `AGENTS.md:18` (R1.1), the `tests/` scan gap and its live counterexample
(R1.6), the vacuity of `assert roots` (R1.7), and the existence of the
`DECISION.md` amendment R1.4 cites.

### Every Done Means is green

| # | Proof | Result |
|---|---|---|
| 1 | `uv sync && uv run python -c "import scufris_core, scufris_host, scufris_hostctl, scufris_hostd, scufris"` | PASS |
| 2 | `test_no_package_imports_a_sibling_private_module` | PASS |
| 3 | `test_package_import_graph_matches_the_declared_graph` | PASS |
| 4 | `test_core_is_domain_free` | PASS |
| 5 | `test_every_package_model_is_registered` | PASS |
| 6 | `test_every_package_has_a_gated_example` | PASS |
| 7 | `rg -q 'packages/\*/tests' pyproject.toml` | PASS |
| 8 | `! rg -q 'legacy_agent\|db/legacy\|db\.legacy\|legacy_import' ...` | PASS |
| 9 | `nix flake check && nix build .#scufris .#scufris-web .#scufris-hostd` | PASS, `result/bin/scufris-hostd` executable |
| 10 | manual | pending, see below |

Full suite: 1108 passed, 1 skipped.

The five guard tests were checked for vacuity and are not vacuous: each has a
falsifier driven through the real checker
(`test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle`,
`test_the_domain_free_check_rejects_the_pre_move_tree`,
`test_the_example_gate_rejects_an_unclaimed_member_and_a_rotted_claim`),
`DECLARED_GRAPH` is compared for EQUALITY so a dead declared edge fails,
`CORE_MODULES` is a real allowlist rather than a property check, and the
metadata side of the model-registration test runs in a fresh subprocess against
`packages/hostctl`'s two real tables.

### Findings

- [x] R1.1 (MAJOR) AGENTS.md:18 - the workspace row describes `packages/core` as
  "the engine, `Database`, `Base` and `logsetup`". The delivered `core` also
  ships `eventbus.py` (`EventBus`) and `supervisor.py` (`Supervisor`,
  `RunState`, `RunPhase`), hoisted by 20260803-214749. The repo's own live agent
  doc understates by half what the package the whole epic is built around
  contains. The enforced guard (`CORE_MODULES`) is correct and green, so nothing
  is unprotected - but the doc an agent reads first is wrong. Change: extend the
  row to name `eventbus` and `supervisor`, matching `scufris/README.md:472`,
  which is already correct.
  - Response: Fixed. The row now names all five modules and points at
    `test_core_is_domain_free` as the allowlist that enforces them, so the doc
    names the guard rather than restating a list that can drift from it again.

- [x] R1.2 (MINOR) tasks/20260803-213242/TASK.md:239 - `20260804-053002` is
  unticked in Child Tasks although its record is `RESOLUTION: DONE` and its
  commit `32cd72f` is on master; `20260804-041340` ("Fix the examples the package
  carve broke", `0879f2d`) - work this epic caused - is absent from the list
  entirely. Change: tick 053002 and add 041340 before closing.
  - Response: Fixed. 053002 is ticked and 041340 is listed above it, ticked, and
    labelled as unplanned work the carve caused rather than a planned child, so
    the list does not read as if the epic foresaw it.

- [x] R1.3 (MINOR) tasks/20260803-213242/TASK.md:145 and :32 - "It is the engine,
  `Database`, `Base` and nothing else" and "That is all of it" are false against
  the delivered `core`, and are already retracted by this epic's own
  `DECISION.md:173` ("Amendment: `core` is no longer sqlalchemy-only"). Change:
  amend both sentences to the shipped five-module allowlist, citing the
  amendment, so the record does not close on a claim it has itself withdrawn.
  - Response: Fixed. The table row lists the shipped modules and points at the
    amendment; the prose paragraph states the plan, states what shipped, cites
    the amendment's reason (`hostctl` supervising its own applies), and keeps
    the smallness claim in the form that survived - a COST per entry enforced by
    `CORE_MODULES`, not a module count.

- [x] R1.4 (MINOR) tasks/20260803-213242/TASK.md:30 - the ten-unit table carries
  no marking that `agents`, `chat`, `flow` and `telegram` were never carved. The
  deferral is real and consistent - stated in the Epic prose, in the open
  host-approval question, and in `DECISION.md` - but the table plus a fully
  ticked checklist reads as ten delivered units when five shipped. Change:
  annotate those four rows as deferred, naming `tasks/20260729-102157` and the
  unanswered host-approval question as the reason.
  - Response: Fixed. The four rows are marked "DEFERRED, not carved", and a
    paragraph under the table states that five of the ten shipped, names both
    reasons with their task IDs, and separates what the table is (the target
    cut) from what the tick marks are about (the five carved into it).

- [x] R1.5 (MINOR) tasks/20260803-213242/TASK.md:251 - Manual Acceptance still
  reads `(pending) 20260803-214746` for a DONE child, and Done Means 10 has no
  acceptance entry at all. Both non-automated proofs are unrecorded in a record
  about to close. Change: record the verdicts, or state explicitly that they are
  carried to the epic's close.
  - Response: Fixed by the second option, which is the only honest one here:
    neither check has been put to the maintainer, and confirming a `manual:`
    proof from the work side is what the pending marker exists to prevent. Both
    entries are now listed, both explicitly carried to the epic's close, and
    214746's is annotated to be judged against the five modules that shipped
    rather than the three it was written against.

- [x] R1.6 (MINOR) tests/test_package_boundaries.py:174 - `_import_roots()` scans
  only `packages/*/src/*` and `scufris/`, so the boundary rule is unenforced in
  test code - and there is a live counterexample: `tests/test_logsetup.py:11`
  does `from scufris_core.logsetup import _RequestIdFilter`, reaching around a
  sibling's facade into a private name, exactly what the test at `:369` forbids
  in shipped source. Change: either extend the scan to `tests/` and
  `packages/*/tests/` (a package's own tests may reach its own internals), or
  state the scope limit in the docstring and fix that one import.
  - Response: Fixed by the first option - a rule enforced only in shipped source
    has a hole the size of the suite. New `_test_roots()` maps each member to
    its test directory, and the private-module test scans source plus tests per
    member, so a package's own tests reaching its own internals stays legal. The
    graph test deliberately does NOT scan tests: `DECLARED_GRAPH` is a claim
    about shipped distributions.
    Extending the scan went red on TWO violations, not the one reported.
    `tests/test_logsetup.py` moved to `packages/core/tests/`, which is where a
    test of `scufris_core.logsetup` belonged anyway - it is now core's own test
    reaching core's own private name. The second was
    `packages/core/tests/test_eventbus.py:7` importing `scufris.agent` for its
    payload types: `core` depends on nothing, so its tests may not depend on the
    root either. `EventBus` is generic over its payload, so the test now defines
    a local `StreamEvent` dataclass - the app's real stream events were never
    what it was testing.

- [x] R1.7 (NIT) tests/test_package_boundaries.py:346 and tests/test_examples.py:148
  - `assert roots` / `assert members` cannot fail: `_import_roots()`
  unconditionally inserts `"scufris"`, so the anti-vacuity guard holds even with
  `packages/` empty. Change: assert the globbed member count instead
  (`len(roots) > 1`, or a non-empty `packages/*/src/*` glob).
  - Response: Fixed. All three call sites (both cited, plus the private-module
    test) assert `len(...) > 1`, so an empty `packages/` now fails the guard
    instead of passing it on the unconditionally inserted root.

- [x] R1.8 (NIT) tests/test_package_boundaries.py:374 - the docstring still says
  "With `core` and the root as the only two members the rule has a single pair to
  police; it earns a red run once a second package is carved out beside `core`".
  Four packages are carved. Change: drop the stale sentence.
  - Response: Fixed. The sentence is gone; the docstring now describes the test
    covering test code as well as source, which is what it actually does after
    R1.6.

### Pending manual checks (do not block)

- Done Means 10: the maintainer names the owning package for a given concern
  from the directory listing alone.
- 20260803-214746's acceptance: `core` is small enough that its contents are
  obvious and does not read as a junk drawer. Note that `core` is now five
  modules, not the three the acceptance was written against.

### Verdict

REQUEST_CHANGES.

Every code-level proof of this epic is green and nothing in the delivered carve
needs changing. The open work is documentation honesty: one shipped repo doc
(`AGENTS.md`) is wrong about what `core` contains, and the epic record is about
to close carrying a retracted claim, an unticked done child, a missing child,
and an unmarked half-delivered unit table. For an epic whose stated deliverable
is that the architecture is "visible in the directory tree" and enforced rather
than claimed, the record and the agent doc ARE part of the deliverable.

### Inspection commands

```sh
uv sync && uv run python -c "import scufris_core, scufris_host, scufris_hostctl, scufris_hostd, scufris"
uv run python -m pytest
ls packages/core/src/scufris_core/
rg -n 'packages/core' AGENTS.md
rg -n 'scufris_core\.' tests/ packages/*/tests/
nix flake check && nix build .#scufris .#scufris-web .#scufris-hostd && test -x result/bin/scufris-hostd
```

## Round 2

- REVIEWER: out-of-context `general-purpose` subagent (id `a5d3e0b9c9eb64188`),
  given only the task ID, the repository path, the branch, the fix commit
  `0c762a8`, the review dimensions and the finding format.
- VERDICT: APPROVE

All eight round-1 findings verified fixed and ticked above. The reviewer
re-derived each Response against the tree; the primary independently re-derived
two of them: the `_facade_problems` call-site probe behind R2.1 (see below) and
the R2.2 miscount, read directly at `TASK.md:336,340`.

### Round-1 fixes, verified

| # | Verdict | Evidence |
|---|---|---|
| R1.1 | CONFIRMED | `AGENTS.md:18` names all five modules and cites `test_core_is_domain_free` as the allowlist; matches `ls packages/core/src/scufris_core/` |
| R1.2 | CONFIRMED | both children ticked; 20260804-041340 present and labelled unplanned (`TASK.md:256-259`) |
| R1.3 | CONFIRMED | `TASK.md:32` and `:153-161` separate planned from shipped and cite the `DECISION.md` amendment, which exists |
| R1.4 | CONFIRMED | four rows marked "DEFERRED, not carved" with reasons, plus the five-of-ten paragraph at `:47` |
| R1.5 | CONFIRMED | both `manual:` proofs listed and explicitly carried; neither self-ticked |
| R1.6 | CONFIRMED | `_test_roots()` plus the per-member scan; a probe import of `scufris_core.engine` planted in `packages/hostd/tests/` turns the real test red; no sibling-submodule import remains in any test directory; `test_logsetup.py` moved and collected; `test_eventbus.py` no longer imports `scufris.agent` |
| R1.7 | CONFIRMED | `len(...) > 1` at `tests/test_package_boundaries.py:369,477` and `tests/test_examples.py:148` |
| R1.8 | CONFIRMED | stale sentence gone; the docstring now describes the tests arm |

### Done Means

Re-run this round: full suite 1109 passed, 1 skipped; `ruff check`,
`ruff format --check`, `mypy` (235 files), `tatr check` clean; `nix flake check`
exit 0. Done Means 1-9 green, unchanged from round 1. Done Means 10 is `manual:`
and stays pending.

The three `nix build` outputs were not re-run this round: nothing in `0c762a8`
touches `flake.nix`, `nix/`, dependency lists or console scripts, and
`nix flake check` is green. Round 1's build evidence stands.

### Findings

- [ ] R2.1 (NIT) tests/test_package_boundaries.py:478 - the R1.6 tests arm is
  enforced but its WIRING is unproven. `_facade_problems`'s falsifier pins the
  arm inside the helper - mutating its loop to source-only turns
  `test_the_facade_check_rejects_a_reach_from_source_and_from_tests` red, as the
  Response claims - but replacing `_facade_problems(roots, _test_roots())` with
  `_facade_problems(roots, {})` at the real call site leaves all six tests in
  the file green (primary-run mutation probe, restored after). A `_test_roots()`
  that silently returns `{}` - a renamed test directory, a member without one -
  reopens the suite-sized hole R1.6 closed, which is the same vacuity class as
  R1.7. Change: assert the map is real in
  `test_no_package_imports_a_sibling_private_module`, e.g.
  `tests = _test_roots(); assert len(tests) > 1 and {"scufris_core", "scufris"} <= set(tests)`.
  - Response:

- [ ] R2.2 (NIT) tasks/20260803-213242/TASK.md:336 - the Close-out miscounts its
  own split: "Five were record and doc honesty (R1.1-R1.5, R1.8)" lists six IDs,
  and "Three were the guard tests themselves (R1.6, R1.7)" lists two. The IDs
  are right and the prose is not, in the paragraph whose job is to state
  honestly what round 1 produced. Change: "Six were record and doc honesty
  (R1.1-R1.5, R1.8) ... Two were the guard tests themselves (R1.6, R1.7)."
  - Response:

Both are NIT and neither blocks the verdict. They are recorded for the fix pass
that any later touch of these files should carry.

- Process signal: a full-suite run during this round failed
  `tests/test_app.py::test_orchestrator_chat_uses_server_cwd`; it passed in
  isolation and on the next full run. Cause is `_wait_state`
  (`tests/test_app.py:2577`), which polls for 2s and then RETURNS the last state
  instead of failing, so a timing loss reads as a state mismatch. Pre-existing
  and outside this epic's diff, so filed as `20260804-112025` rather than
  reviewed here.
- `examples/` is scanned by neither the facade rule nor the graph rule (source
  plus per-member tests only). Clean today, outside R1.6's scope; noted as the
  remaining unscanned tree.

### Pending manual checks (do not block)

Unchanged from round 1, both still unconfirmed by the maintainer and carried to
this epic's close:

- Done Means 10: the maintainer names the owning package for a given concern
  from the directory listing alone.
- 20260803-214746's acceptance: `core` is small enough that its contents are
  obvious and does not read as a junk drawer - judged against the five modules
  that shipped, not the three the child was written against.

### Inspection commands

```sh
uv run python -m pytest
uv run ruff check && uv run ruff format --check . && uv run mypy
tatr check && nix flake check
rg -n '_test_roots|_facade_problems' tests/test_package_boundaries.py
sed -n '330,345p' tasks/20260803-213242/TASK.md
```
