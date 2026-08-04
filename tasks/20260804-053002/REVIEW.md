# Review: Prove the declared dependency graph and the example gate

- TASK: 20260804-053002
- BRANCH: test/prove-declared-graph-and-example-gate

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tests/test_package_boundaries.py:294 - the cycle arm of
  `_graph_problems` (line 278) is never exercised THROUGH the checker.
  `assert _cycles({...})` pins the helper alone, so deleting the
  `for cycle in _cycles(declared)` loop from `_graph_problems` leaves both new
  tests green - confirmed by mutation on this branch. DoD criterion 2 says the
  CHECKER is proven to bite on a cycle, so add to the falsifier:
  `assert [p for p in _graph_problems({"a": frozenset({"b"}), "b": frozenset({"a"})}, {"a": {"b": ["x.py"]}, "b": {"a": ["y.py"]}}) if p.startswith("cycle in the declared graph")]`.
  - Response: Fixed. The falsifier now drives the cycle arm through
    `_graph_problems` on exactly that input, and the two direct `_cycles`
    asserts are gone - every arm is reached through the checker or not at all.
    Mutation-confirmed: blanking the `for cycle in _cycles(declared)` loop and,
    separately, the `if node in stack` detection inside `_cycles` each turn the
    falsifier red.
- [x] R1.2 (MAJOR) tests/test_package_boundaries.py:292 - three of the four arms
  Step 1 names are unasserted. The dead-edge arm (line 273) fires into the
  falsifier's `problems` but nothing asserts it, and both member-set-mismatch
  arms (lines 256, 261) are unreachable from any test. Deleting the dead-edge
  arm leaves `tests/test_package_boundaries.py` and `tests/test_examples.py`
  green - confirmed by mutation. That arm is the equality half NOTES.md chose
  over containment, so it is the one most worth pinning. Extend the falsifier
  with `assert [p for p in problems if "declared but nothing imports it" in p], problems`,
  plus one `_graph_problems` call whose `edges` carries an extra key
  (`"scufris_agents": {}`) and one with a key dropped, asserting each mismatch
  message.
  - Response: Fixed. The falsifier asserts the dead-edge message, adds a
    `"scufris_agents": {}` key for the on-disk-not-declared arm and drops
    `"scufris"` for the declared-not-on-disk arm. All four arms plus the
    disallowed-edge arm are mutation-confirmed red one at a time. The same
    charge applied to `tests/test_examples.py`, whose arms no test drove
    either, so its body became `_example_problems(members, claimed)` with its
    own falsifier,
    `test_the_example_gate_rejects_an_unclaimed_member_and_a_rotted_claim`;
    its five arms are mutation-confirmed too.
- [x] R1.3 (MAJOR) AGENTS.md:73 - the "New workspace member" procedure still
  lists only `uv lock` plus re-entering `nix develop`. Carving a member now also
  requires a `DECLARED_GRAPH` entry in `tests/test_package_boundaries.py` and an
  `EXAMPLES_BY_MEMBER` entry in `tests/test_examples.py` naming an example that
  is on `OFFLINE` and imports the member, or the suite goes red with no pointer
  to why. Append both to that bullet.
  - Response: Fixed. The "New workspace member" bullet now names both
    literals and what each entry must satisfy.
- [x] R1.4 (MINOR) scufris/README.md:465 - "`tests/test_package_boundaries.py`
  enforces both that rule and the rule that `core` stays generic" now enumerates
  two of the three rules that module enforces. Name the third: `DECLARED_GRAPH`
  is checked for equality against the real sibling imports and for acyclicity.
  - Response: Fixed. The sentence now enumerates three rules and names the
    equality and acyclicity checks on `DECLARED_GRAPH`.
- [x] R1.5 (MINOR) AGENTS.md:23 - the sources-of-truth row calls
  `pyproject.toml`/`uv.lock` "the ONE place membership is declared", which the
  diff makes false: two test literals must now agree with it. Reword to name
  `uv.lock` as the build's declaration and `DECLARED_GRAPH` /
  `EXAMPLES_BY_MEMBER` as mirrors that fail on drift.
  - Response: Fixed. The row calls `uv.lock` the build's declaration and
    both test literals mirrors that go red on drift.
- [x] R1.6 (MINOR) tests/test_examples.py:71 -
  `assert set(EXAMPLES_BY_MEMBER) == members` short-circuits before the
  `problems` loop, so a member-set mismatch hides every other failure and the
  message is a bare set diff. This contradicts the report-all-arms style of
  `_domain_free` and `_graph_problems` in the sibling module. Fold the mismatch
  into `problems` - one message per unclaimed member and per stale key - and
  keep the single `assert problems == []`.
  - Response: Fixed. Both mismatch directions are messages in `problems`,
    the per-member loop skips members that are not on disk, and the test is one
    `assert _example_problems(members, EXAMPLES_BY_MEMBER) == []`.
- [ ] R1.7 (NIT) tests/test_package_boundaries.py:230 - `walk` enumerates every
  simple path with no memo of nodes already proven acyclic, so cost is
  exponential in a dense graph. Harmless at five nodes; the epic's graph reaches
  nine. Add a `done: set[str]` of fully-explored nodes and return early when
  `node in done`.
  - Response: Fixed. `done` is added, checked after the stack test and set
    after the neighbour loop; the docstring records why skipping an explored
    node loses no cycle.

Process signal: the Close-out's Evidence paragraph is accurate on every claim it
makes but picks claims that step around the weakest part of the change. "The
falsifier was proven to bite" is true of `_graph_problems` as a whole and of
`_cycles` as a helper, and reads as coverage of all four arms without ever
saying so. A per-arm mutation list would have surfaced R1.1 and R1.2 during
work.

Round 1 was delegated to a reviewer with no sight of the implementing session.
The recording pass re-ran every check and re-derived the load-bearing claims
independently.

Verified in-session: the sprout `.venv` resolves all five import roots from the
worktree. All four DoD commands exit 0. The full suite's progress characters
count 1107 `.` and one `s`, so the Close-out's "1107 passed, 1 skipped" is
exact; `wc -l` gives 346 and 121, also exact, against the 900-line cap.
`ruff check`, `ruff format --check`, `mypy` on both files,
`scripts/check_file_size.py` and `tatr -r . check` all exit 0. The real edges
measured by `_sibling_edges` equal `DECLARED_GRAPH` exactly. Stubbing
`_graph_problems` to `[]` and stubbing `_cycles` to `[]` each turn the falsifier
red, as the Close-out claims. All four arms of `_graph_problems` were driven by
hand and each emits its message. The mutations behind R1.1 and R1.2 were run
here, not taken on report.

Not a finding, pre-existing: `uv run pytest tests/` (a path argument rather than
the configured `testpaths`) fails collection with
`ModuleNotFoundError: No module named 'test_host_actions'` from
`tests/conftest.py:291`. It predates this branch and is worth its own task.
Filed as 20260804-101727; not fixed here.

Not verified: `nix flake check`, `nix build` and the frontend gate were not run.
The epic's other eight Done Means were not re-checked; this diff touches nothing
they cover.

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R2.1 (MINOR) tests/test_package_boundaries.py:241 - the `done` memo added
  for R1.7 silently drops cycles, and both docstring claims about it are false.
  "Every cycle reachable in `graph`" (line 221) and "Skipping them loses no
  cycle" (lines 229-232) do not hold: on
  `{"b": {"c", "x"}, "c": {"m"}, "m": {"b"}, "x": {"m"}}` the memoised helper
  reports only `['b', 'c', 'm', 'b']` and never `['b', 'x', 'm', 'b']`, because
  `m` is marked done inside the first branch and the second reaches it through
  `x`. Existence detection is unaffected - a node goes into `done` only after
  full exploration, which is textbook DFS cycle detection - so `_graph_problems`
  still bites, and the arm-level mutations stay red. But the branch is covered
  by no test (deleting `if node in done: return` leaves both files green), and
  this is the same "asserted in prose rather than driven by a test" pattern that
  R1.1 and R1.2 punished. Delete the memo: `done: set[str] = set()`,
  `if node in done: return`, `done.add(node)` and the docstring paragraph that
  justifies them. `DECLARED_GRAPH` is five nodes and the epic's graph is nine,
  so the asymptotics R1.7 raised buy nothing at this size, and R1.7 was a NIT.
  If the memo is kept instead, the docstring must claim one cycle per DFS tree
  rather than every cycle, and a test must drive the difference.
  - Response: Fixed, by deletion. The memo, its two `walk` lines and the
    docstring paragraph are gone, and the docstring now says why a memo is
    refused rather than claiming one is free. The property the memo broke is
    pinned: the falsifier asserts `_graph_problems` reports exactly TWO cycle
    messages for the counterexample graph, so re-adding the exact memo turns
    the suite red (mutation-confirmed). R1.7 is answered by this: its
    optimisation is declined with reason, not taken.
- [x] R2.2 (MINOR) tests/test_package_boundaries.py:1 - the module docstring
  still opens "The carve's two claims, as checks instead of README paragraphs"
  and enumerates two, while this diff gives the module a third it now owns:
  `DECLARED_GRAPH` checked for equality against the real sibling imports and for
  acyclicity. `scufris/README.md:465` was corrected to three rules for R1.4 and
  the module's own header was missed by the same sweep. Say three claims in the
  first line and add a paragraph for the graph beside the existing
  `test_core_is_domain_free` and falsifier paragraphs.
  - Response: Fixed. The first line says three claims, the opening paragraph
    names the dependency direction beside the other two, and a new paragraph
    covers `DECLARED_GRAPH`, its equality-not-containment choice and its
    division of labour with
    `test_no_package_imports_a_sibling_private_module`.
- [x] R2.3 (NIT) tests/test_examples.py:106 - `_example_problems` reads `OFFLINE`
  and `EXAMPLES` from module globals, so its falsifier has to borrow a real
  example (`telegram_bot.py`) to drive the off-`OFFLINE` arm; adding that script
  to `OFFLINE` later turns the falsifier red for a reason unrelated to the arm it
  tests. The sibling module already sets the precedent - `_domain_free(package_root,
  allowed)` takes its allowlist as a parameter precisely so its falsifier can pass
  a different one. Give `_example_problems` an `offline` parameter and pass a
  hand-built tuple from the falsifier.
  - Response: Fixed. `_example_problems(members, claimed, offline=OFFLINE)`
    takes the list as a parameter, mirroring `_domain_free(package_root,
    allowed)`. The falsifier no longer borrows `telegram_bot.py`: it drives the
    off-list arm with `offline=()` and the other two arms with a one-element
    tuple, so no arm depends on which real scripts are on `OFFLINE`.

Process signal: R1.7 was a NIT that asked for an optimisation, and taking it
introduced the round's only finding - a behaviour change justified by a docstring
nobody could fail. Round 2's Evidence paragraph fixed the shape R1.1/R1.2
punished for the checker arms, then the same shape reappeared one function over,
in the one change that was not driven by a falsifier. A NIT that changes what a
helper returns is not a NIT.

Round 2 was delegated to a reviewer with no sight of the implementing session.
The recording pass re-derived the load-bearing claim rather than taking it on
report: the counterexample graph was run against the memoised helper and against
an un-memoised copy in the same process, giving one cycle versus two, with
existence agreeing; and `_graph_problems` on that graph reports one cycle
message, which is why the gate still bites. Deleting `if node in done: return`
and running both test files was confirmed GREEN here, so R2.1's coverage claim
holds.

Verified in-session: worktree clean before and after every mutation. Full suite
1108 passed, 1 skipped, exit 0. `tatr -r . check` exits 0. Round 1's R1.1 through
R1.6 are confirmed fixed against the diff and ticked; R1.7 is implemented but
defective, which R2.1 owns. The pre-existing `uv run pytest tests/` collection
failure is filed as 20260804-101727 and is not this branch's problem.

Not verified: `nix flake check`, `nix build` and the frontend gate. The epic's
other Done Means were not re-checked; this diff touches nothing they cover.

## Round 3

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [x] R3.1 (NIT) tests/test_examples.py:1 - the module docstring is the one doc
  surface the R2.2 sweep missed. It still describes the file as only running the
  examples and explaining `OFFLINE`, while the diff gives the module a second
  claim it now owns: `EXAMPLES_BY_MEMBER` proves every workspace member names an
  example that is on `OFFLINE`, exists and imports it, which is what makes the
  opt-in tuple more than a hand-written list. Same omission R2.2 charged against
  `tests/test_package_boundaries.py:1`, one module over. Add a paragraph naming
  `EXAMPLES_BY_MEMBER`, the member set imported from `test_package_boundaries`,
  and why the claim is declared rather than inferred from what an example
  imports.
  - Response: Fixed in e398268. The paragraph names all three, and gives
    `hostctl_approval_flow.py` importing all four packages as the reason
    inference was refused.
- [x] R3.2 (NIT) tests/test_package_boundaries.py:231 - "one representative path
  is reported per set of mutually reachable nodes" contradicts the assertion
  added 90 lines below. On the falsifier's own graph
  (`b -> {c, x}, c -> {m}, x -> {m}, m -> b`) all four nodes are mutually
  reachable, yet the falsifier asserts exactly TWO cycle messages. The dedup key
  is `frozenset(cycle)`, not the SCC. Since R2.1 was a false docstring claim
  about this exact function's return, tighten it to "per set of nodes that forms
  a cycle rather than every rotation of it". The paragraph below it, on the
  refused memo, is accurate and should stay.
  - Response: Fixed in e398268. The docstring now says the key is the cycle's
    node set, states explicitly that it is not per strongly connected
    component, and points at the four-node graph as the case that proves it.

Both findings are NITs and neither blocked the verdict. They were taken and
committed BEFORE this round was written, and the boxes are ticked on that basis
rather than on a later round: R3.2 is the same class of defect as R2.1 - a
docstring claiming a property no test drives - which is the one thing this
branch exists to argue against, so leaving it open for the sake of round
symmetry would have been the wrong trade.

Process signal (from the round-3 reviewer, no adverse finding): the R1.7 -> R2.1
chain closed the right way. The optimisation was declined with a stated reason
rather than documented down, and the property it broke is pinned by an assertion
rather than by prose, which is the standard rounds 1 and 2 set.

Round 3 was delegated to a reviewer with no sight of the implementing session.
It reproduced every number in all three close-outs exactly - 389/176 lines at its
HEAD, 372/165 at round 2, 346/121 at round 1 - and ran thirteen mutations one at
a time, all thirteen red: the six `_graph_problems`/`_cycles` arms, the five
`_example_problems` arms, the faithful re-insertion of the `done` memo, and the
head filter in `_sibling_edges`. It confirmed R2.1, R2.2 and R2.3 fixed and
R1.1-R1.6 not regressed, and confirmed `git status --short` empty before and
after every mutation.

The recording pass re-derived R3.2 rather than taking it on report: `_cycles` on
the four-node graph returns two paths whose node sets differ, while the four
nodes are one mutually reachable set, so the old wording was false in the same
direction R2.1 was. After the two fixes: full suite 1108 passed, 1 skipped,
exit 0; `ruff check`, `ruff format --check`, `mypy` on both files,
`scripts/check_file_size.py` and `tatr -r . check` exit 0; 391 and 185 lines
against the 900-line cap. The changes are comment-only, so the mutation results
above stand.

Pending user checks: none. The task carries no `manual:` proof.

Not verified in any round: `nix flake check`, `nix build` and the frontend gate.
The epic's other eight Done Means were not re-checked; this diff touches nothing
they cover.
