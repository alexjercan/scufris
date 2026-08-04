# Prove the declared dependency graph and the example gate

- PRIORITY: 100
- TAGS: architecture, packaging, tests
- KIND: STORY
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
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

- [ ] Add the graph machinery to `tests/test_package_boundaries.py`, below
      `_import_roots()` and reusing it: the `DECLARED_GRAPH` literal from
      NOTES.md, `_sibling_edges(roots)` (member -> imported member -> importing
      modules, built from `_modules()` and `_imported_modules()`),
      `_cycles(graph)`, and `_graph_problems(declared, edges)` with its four
      arms - member set mismatch, disallowed real edge naming the importers,
      declared edge with no real import, cycle in `declared`.
- [ ] Add the falsifier
      `test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle` against
      hand-built `edges` (a `scufris_core -> scufris_host` edge) and a
      hand-built two-node cycle. Written before the green test so a
      `_graph_problems` that returns `[]` unconditionally cannot pass.
- [ ] Add `test_package_import_graph_matches_the_declared_graph`:
      `_graph_problems(DECLARED_GRAPH, _sibling_edges(_import_roots())) == []`.
      Planning confirmed the real edges already equal `DECLARED_GRAPH`, so this
      is green on first run; the falsifier is what proves it can go red.
- [ ] Add `EXAMPLES_BY_MEMBER` (NOTES.md literal) and
      `test_every_package_has_a_gated_example` to `tests/test_examples.py`:
      import `_import_roots` and `_imported_modules` from
      `test_package_boundaries` (same `tests/` dir, on `sys.path` under the
      repo's pytest config), then assert key set == member set, each name is in
      `OFFLINE`, the file exists, and the example imports the member it claims.
- [ ] Run the four DoD commands plus `tatr check`, and confirm the two test
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
