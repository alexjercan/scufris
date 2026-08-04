# Prove the declared dependency graph and the example gate

- PRIORITY: 100
- TAGS: architecture, packaging, tests
- KIND: STORY
- ACTIVITY: PLANNING
- GATES: -
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

## Done Means

1. The declared dependency graph is asserted against the real imports, and a
   member importing a sibling the graph does not allow fails
   (test: `test_package_import_graph_matches_the_declared_graph`).
2. The graph is proven acyclic by the same test, not by reading it.
3. Every workspace member is proven to have an example on the offline gate's
   opt-in list, so a package that never adds itself cannot leave the gate green
   (test: `test_every_package_has_a_gated_example`).
4. The whole suite is green and the epic's other proofs are unaffected
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
