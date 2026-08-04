# Review: 20260803-213242

## Round 1

- Reviewer: out-of-context `general-purpose` subagent (id `a480234221fe27c62`),
  given only the task ID, the repository path, the review dimensions and the
  finding format.
- Primary re-derived, independently of the reviewer: the `core` module listing
  against `AGENTS.md:18` (finding 1), the `tests/` scan gap and its live
  counterexample (finding 6), the vacuity of `assert roots` (finding 7), and
  the existence of the `DECISION.md` amendment finding 4 cites.
- Scope: no feature branch. The epic's six children have all landed on
  `master`, so the subject is the cumulative tree, judged against the epic's
  own Done Means.

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

**1. MAJOR - `AGENTS.md:18`.** The workspace row describes `packages/core` as
"the engine, `Database`, `Base` and `logsetup`". The delivered `core` also
ships `eventbus.py` (`EventBus`) and `supervisor.py` (`Supervisor`, `RunState`,
`RunPhase`), hoisted by 20260803-214749. The repo's own live agent doc
understates by half what the package the whole epic is built around contains.
The enforced guard (`CORE_MODULES`) is correct and green, so nothing is
unprotected - but the doc an agent reads first is wrong.
Change: extend the row to name `eventbus` and `supervisor`, matching
`scufris/README.md:472`, which is already correct.

**2. MINOR - `tasks/20260803-213242/TASK.md:239`.** `20260804-053002` is
unticked in Child Tasks although its record is `RESOLUTION: DONE` and its
commit `32cd72f` is on master; `20260804-041340` ("Fix the examples the package
carve broke", `0879f2d`) - work this epic caused - is absent from the list
entirely. Change: tick 053002 and add 041340 before closing.

**3. MINOR - `tasks/20260803-213242/TASK.md:145` and `:32`.** "It is the
engine, `Database`, `Base` and nothing else" and "That is all of it" are false
against the delivered `core`, and are already retracted by this epic's own
`DECISION.md:173` ("Amendment: `core` is no longer sqlalchemy-only"). Change:
amend both sentences to the shipped five-module allowlist, citing the
amendment, so the record does not close on a claim it has itself withdrawn.

**4. MINOR - `tasks/20260803-213242/TASK.md:30`.** The ten-unit table carries
no marking that `agents`, `chat`, `flow` and `telegram` were never carved. The
deferral is real and consistent - stated in the Epic prose, in the open
host-approval question, and in `DECISION.md` - but the table plus a fully
ticked checklist reads as ten delivered units when five shipped. Change:
annotate those four rows as deferred, naming `tasks/20260729-102157` and the
unanswered host-approval question as the reason.

**5. MINOR - `tasks/20260803-213242/TASK.md:251`.** Manual Acceptance still
reads `(pending) 20260803-214746` for a DONE child, and Done Means 10 has no
acceptance entry at all. Both non-automated proofs are unrecorded in a record
about to close. Change: record the verdicts, or state explicitly that they are
carried to the epic's close.

**6. MINOR - `tests/test_package_boundaries.py:174`.** `_import_roots()` scans
only `packages/*/src/*` and `scufris/`, so the boundary rule is unenforced in
test code - and there is a live counterexample: `tests/test_logsetup.py:11` does
`from scufris_core.logsetup import _RequestIdFilter`, reaching around a
sibling's facade into a private name, exactly what the test at `:369` forbids
in shipped source. Change: either extend the scan to `tests/` and
`packages/*/tests/` (a package's own tests may reach its own internals), or
state the scope limit in the docstring and fix that one import.

**7. NIT - `tests/test_package_boundaries.py:346` and
`tests/test_examples.py:148`.** `assert roots` / `assert members` cannot fail:
`_import_roots()` unconditionally inserts `"scufris"`, so the anti-vacuity
guard holds even with `packages/` empty. Change: assert the globbed member
count instead (`len(roots) > 1`, or a non-empty `packages/*/src/*` glob).

**8. NIT - `tests/test_package_boundaries.py:374`.** The docstring still says
"With `core` and the root as the only two members the rule has a single pair to
police; it earns a red run once a second package is carved out beside `core`".
Four packages are carved. Change: drop the stale sentence.

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
rg -n 'scufris_core\.' tests/
nix flake check && nix build .#scufris .#scufris-web .#scufris-hostd && test -x result/bin/scufris-hostd
```
