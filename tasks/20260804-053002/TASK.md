# Prove the declared dependency graph and the example gate

- PRIORITY: 100
- TAGS: architecture, packaging, tests
- KIND: STORY
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

Two of the epic's ten Done Means name a test that does not exist. Every other
proof is green on master after the five carve children landed, so these two are
all that stands between the workspace and a closed epic:

- `test_package_import_graph_matches_the_declared_graph` (Done Means 3) - the
  epic's `### Dependency direction` graph lives only in prose. Nothing checks
  that `host` imports no sibling, that `hostd` reaches only `core` and `host`,
  or that the real import edges stay acyclic. `test_no_package_imports_a_sibling_private_module`
  polices HOW a sibling is imported, never WHICH siblings may be imported at
  all, so a new edge - `core -> host`, say - lands green today.
- `test_every_package_has_a_gated_example` (Done Means 6) - `tests/test_examples.py`
  `OFFLINE` is a hand-written opt-in tuple. A package that ships no example, or
  ships one and never adds it to the tuple, leaves the gate green. The epic's
  proof rule is that a package is done because its example runs, so the opt-in
  list has to be proven to cover every workspace member.

Both are tests only. No production code moves.

The declared graph must be written down in ONE place the test reads - a literal
in the test module - and it covers the four members that exist plus the root
(`core`, `host`, `hostd`, `hostctl`, `scufris`). The four packages the epic
lists but does not carve (`agents`, `chat`, `flow`, `telegram`) are out of
scope; the graph literal grows when they do.

## Steps

- [x] Add the graph machinery to `tests/test_package_boundaries.py`, below
      `_import_roots()` and reusing it: the `DECLARED_GRAPH` literal from
      NOTES.md, `_sibling_edges(roots)` (member -> imported member -> importing
      modules, built from `_modules()` and `_imported_modules()`),
      `_cycles(graph)`, and `_graph_problems(declared, edges)` with its four
      arms - member set mismatch, disallowed real edge naming the importers,
      declared edge with no real import, cycle in `declared`.
- [x] Add the falsifier
      `test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle` against
      hand-built `edges` (a `scufris_core -> scufris_host` edge) and a
      hand-built two-node cycle. Written before the green test so a
      `_graph_problems` that returns `[]` unconditionally cannot pass.
- [x] Add `test_package_import_graph_matches_the_declared_graph`:
      `_graph_problems(DECLARED_GRAPH, _sibling_edges(_import_roots())) == []`.
      Planning confirmed the real edges already equal `DECLARED_GRAPH`, so this
      is green on first run; the falsifier is what proves it can go red.
- [x] Add `EXAMPLES_BY_MEMBER` (NOTES.md literal) and
      `test_every_package_has_a_gated_example` to `tests/test_examples.py`:
      import `_import_roots` and `_imported_modules` from
      `test_package_boundaries` (same `tests/` dir, on `sys.path` under the
      repo's pytest config), then assert key set == member set, each name is in
      `OFFLINE`, the file exists, and the example imports the member it claims.
- [x] Run the four DoD commands plus `tatr check`, and confirm the two test
      files stay under the 900-line test cap
      (`scripts/check_file_size.py`, exercised by
      `test_check_file_size_passes_on_the_repository`).

## Definition of Done

- The declared graph is asserted against the real imports, so a member
  importing a sibling the graph does not allow fails
  (cmd: `uv run pytest -q
  "tests/test_package_boundaries.py::test_package_import_graph_matches_the_declared_graph"`).
- The checker is proven to bite on both an undeclared edge and a cycle, rather
  than being trusted because it returns `[]`
  (cmd: `uv run pytest -q
  "tests/test_package_boundaries.py::test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle"`).
- Every workspace member `_import_roots()` finds names an example that is on
  `OFFLINE`, exists, and imports that member, so a package that never adds
  itself cannot leave the gate green
  (cmd: `uv run pytest -q
  "tests/test_examples.py::test_every_package_has_a_gated_example"`).
- The whole suite is green and the epic's other proofs are unaffected
  (cmd: `uv run pytest -q`).

## Notes

- Epic: tasks/20260803-213242 - Done Means 3 and 6.
- `tests/test_package_boundaries.py` already has `_import_roots()`,
  `_modules()` and `_imported_modules()`; the graph test is a consumer of them,
  not a new AST walker.
- `tests/test_examples.py:32` is the `OFFLINE` tuple. Today it names
  `core_unit_of_work.py`, `host_report_fixture.py`, `hostctl_approval_flow.py`
  and `hostd_socket_roundtrip.py` - one per carved package, so the mapping
  member -> example is not one the filename gives away for free (`host`'s
  example is `host_report_fixture.py`, not `host_inspect.py`). Decide how a
  member claims its example during understanding.
- Answered in NOTES.md: a member claims its example through a declared
  `EXAMPLES_BY_MEMBER` literal, not an inferred filename rule.
  `hostctl_approval_flow.py` imports all four packages, so "an example covers
  what it imports" would cover the workspace on its own.
- Planning measured the real edges with `_sibling_edges`' logic in scratch and
  they equal `DECLARED_GRAPH` exactly today: `scufris_core` and `scufris_host`
  import no sibling, `scufris_hostd -> core, host`,
  `scufris_hostctl -> core, host, hostd`, `scufris -> all four`. The equality
  arm of the checker is therefore satisfiable as written; no import moves.
- Example imports, also measured: `core_unit_of_work.py` imports
  `scufris_core`, `host_report_fixture.py` `scufris_host`,
  `hostd_socket_roundtrip.py` `scufris_host` and `scufris_hostd`,
  `hostctl_approval_flow.py` all four, `host_agent.py` `scufris`,
  `scufris_hostctl`, `scufris_hostd`. Every claim in `EXAMPLES_BY_MEMBER`
  holds.
- Both DoD test IDs are red on master today - `pytest` exits with
  `ERROR: not found` for each, which is the intended missing change.
- Cap headroom: `tests/test_package_boundaries.py` is 224 lines and
  `tests/test_examples.py` 74, against a 900-line test cap.

## Close-out

What and why. `tests/test_package_boundaries.py` gains `DECLARED_GRAPH`,
`_sibling_edges`, `_cycles`, `_graph_problems` and the two graph tests;
`tests/test_examples.py` gains `EXAMPLES_BY_MEMBER` and
`test_every_package_has_a_gated_example`. No production code moved, exactly as
planned. The epic's dependency direction and its "a package is done because its
example runs" rule were prose; both are now literals a test reads, so the next
carve pays the declaration cost or goes red.

Alternatives. Containment instead of equality for the graph was rejected in
NOTES.md and stayed rejected: a declared edge whose last import disappears must
fail, or the declaration drifts into fiction. Inferring a member's example from
what the example imports was also rejected - `hostctl_approval_flow.py` imports
all four packages and would cover the workspace on its own.

Difficulties and diagnosis. Three, all small. The sprout's `.venv` resolved
`scufris` to the main tree and had no `scufris_core`; `uv sync --all-packages`
in the worktree fixed it, and the DoD IDs then reported the intended
`ERROR: not found` rather than a collection error. `ruff` reordered the new
`from test_package_boundaries import ...` line and reformatted one comprehension.
`mypy` wanted an annotation on the falsifier's hand-built `edges` dict.

Evidence. The falsifier was proven to bite before the machinery was trusted: with
`_graph_problems` stubbed to `[]` it fails, and with `_cycles` stubbed to `[]` it
fails (scratch run, both red). The example gate was probed the same way - dropping
a member from `EXAMPLES_BY_MEMBER`, pointing one at a file outside `OFFLINE`, and
pointing `scufris_core` at `host_report_fixture.py` each turn it red. All four DoD
commands exit 0. Full suite: 1107 passed, 1 skipped. `ruff check`, `ruff format
--check`, `mypy` on both files, `scripts/check_file_size.py` and `tatr check` are
clean. The files are 346 and 121 lines against the 900-line cap.

Reflection. `_cycles` dedupes by node set rather than by rotation, so a graph with
two distinct cycles over the same nodes reports one path. That is enough for the
caller - the question is whether a cycle exists and who is in it - and the
docstring says so rather than leaving it to be discovered. Worth remembering for
the next task in this repo: a fresh sprout needs `uv sync --all-packages` before
any proof command means what it appears to mean.

## Round 2 (review feedback)

What and why. R1.1 and R1.2 were right and they were the same defect: an arm
asserted through its HELPER, or not asserted at all, is an arm that can be
deleted from the checker with nothing going red. The falsifier now reaches
every arm of `_graph_problems` through `_graph_problems` - the dead-edge
message, a two-node cyclic `declared`, an extra `edges` key and a dropped one -
and the two direct `_cycles` asserts are gone rather than kept as belt and
braces, because a passing helper is exactly what made the gap invisible.

The same charge applied to `tests/test_examples.py`, which the findings did not
make: after folding the member-set mismatch into `problems` for R1.6, that gate
had five unasserted arms and no falsifier. Its body is now
`_example_problems(members, claimed)`, mirroring `_graph_problems`, with
`test_the_example_gate_rejects_an_unclaimed_member_and_a_rotted_claim` driving
unclaimed member, stale key, off-`OFFLINE`, missing file and does-not-import.
`test_every_package_has_a_gated_example` keeps its name and its DoD command.
R1.3, R1.4 and R1.5 are doc edits; R1.7 adds the `done` memo to `walk`.

Difficulties and diagnosis. The first mutation harness restored each mutated
file with `git checkout -- <path>`, which threw away the uncommitted fixes it
was supposed to be testing and reported four arms GREEN. The tell was that only
the first mutation in the list ever came back red. Rewritten to snapshot the
file text in memory and restore from that; never `git checkout` a file with
unstaged work in it.

Evidence. Per-arm mutation, one at a time, each run against the whole of both
files: graph cycle loop, `_cycles`'s own `if node in stack`, dead-edge,
disallowed-edge, declared-not-on-disk, on-disk-not-declared, and all five
example arms - eleven mutations, eleven red. Full suite 1108 passed, 1 skipped
(exit 0). The four DoD commands and the new falsifier each exit 0. `ruff
check`, `ruff format --check`, `mypy` on both files,
`scripts/check_file_size.py` and `tatr -r . check` clean. 372 and 165 lines
against the 900-line cap.

Reflection. The process signal in the review was fair: "the falsifier was
proven to bite" was a true sentence chosen over the per-arm list that would
have exposed the hole. A checker that returns a list of messages should be
verified message by message, and the cheap way to do that is a mutation loop
kept in the task rather than an argument in the close-out.

## Round 3 (review feedback)

What and why. R2.1 was right and it was mine: the `done` memo taken for R1.7
changed what `_cycles` returns, and I justified it with a docstring paragraph
that was simply false. On `b -> {c, x}, c -> {m}, x -> {m}, m -> b` the memo
reports one cycle and drops the other, because `m` is marked done inside the
first branch. The memo is deleted rather than documented down: R1.7 was a NIT
about asymptotics over a five-node graph the epic grows to nine, and complete
reporting is worth more than the win. What makes this answered rather than
merely reverted is that the property is now pinned - the falsifier asserts TWO
cycle messages for that graph, so re-adding the exact memo goes red.

R2.2 corrects the module docstring to three claims. R2.3 gives
`_example_problems` an `offline` parameter, mirroring `_domain_free`'s
`allowed`, so the falsifier stops borrowing `telegram_bot.py` and no arm
depends on which real scripts sit on `OFFLINE`.

Difficulties and diagnosis. The first attempt to prove the memo was pinned
wrote the mutation as a function attribute rather than the closure variable the
original used, and came back GREEN. The mutation was wrong, not the test: a
faithful re-insertion of the deleted lines is red. A mutation that does not
reproduce the original code proves nothing about it, which is worth more
suspicion than a green result deserves.

Evidence. Twelve per-arm mutations red (six graph, five example, plus faithful
re-insertion of the memo). Full suite 1108 passed, 1 skipped, exit 0. The three
test-level DoD commands, `ruff check`, `ruff format --check`, `mypy` on both
files, `scripts/check_file_size.py` and `tatr -r . check` all exit 0. 389 and
176 lines against the 900-line cap.

Reflection. The round-2 process signal named the pattern exactly: every arm of
the checkers was driven by a falsifier, and the one change that was not - a
NIT-sized optimisation - is the one that shipped a defect. A change that alters
what a function RETURNS is not a NIT, whatever its diff size, and the honest
test for "is this a nit" is whether an existing assertion would notice.

## Round 4 (review feedback)

What and why. Two NITs, both docstrings, both taken. R3.2 was the same class of
defect as R2.1 and worth taking for that reason alone: `_cycles` claimed to
report "one representative path per set of mutually reachable nodes", which its
own falsifier contradicts - the four-node graph is one mutually reachable set
and the test asserts TWO messages. The dedup key is the cycle's node set, not
the SCC, and the docstring now says so. R3.1 gives `tests/test_examples.py` the
`EXAMPLES_BY_MEMBER` paragraph its module docstring never gained, matching what
R2.2 did for the sibling module.

Evidence. Full suite 1108 passed, 1 skipped, exit 0. `ruff check`, `ruff format
--check`, `mypy` on both files, `scripts/check_file_size.py` and `tatr -r .
check` exit 0. 391 and 185 lines against the 900-line cap. Comment-only changes,
so the round-3 mutation results stand unchanged.

Reflection. Both NITs were docstrings describing code the branch added, and both
were wrong in the direction of claiming more than the code delivers. Three
rounds running, the finding that mattered was a comment asserting a property no
test drove. In a change whose whole subject is "an unproven claim is worth
nothing", the docstrings deserved the same scrutiny as the assertions.
