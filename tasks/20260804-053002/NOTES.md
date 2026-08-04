# Notes: Prove the declared dependency graph and the example gate

## What changes

Nothing an operator sees. Two properties that are prose today become failing
tests:

- Before: the epic's `### Dependency direction` block is a README paragraph. A
  new `core -> scufris_host` import lands green, because
  `test_no_package_imports_a_sibling_private_module` polices HOW a sibling is
  imported (facade, not internals), never WHICH siblings may be imported.
  After: the graph is a literal in the test module, checked against the real
  import edges in both directions, and proven acyclic by walking it.
- Before: `tests/test_examples.py` `OFFLINE` is a hand-written opt-in tuple with
  no relation to the member list. A carved package that ships no example, or
  ships one and forgets the tuple, leaves the gate green. After: every workspace
  member `_import_roots()` finds must name its example, that example must be on
  `OFFLINE`, exist, and actually import the member it claims.

Both new tests are red the moment a sixth member appears under `packages/` and
is not declared. That is the point: the declaration is a cost the next carve
pays deliberately.

## Surfaces

| File | Why |
|---|---|
| `tests/test_package_boundaries.py` | gains `DECLARED_GRAPH`, an edge extractor, a cycle finder, the graph test and its falsifier; reuses `_import_roots()`, `_modules()`, `_imported_modules()` unchanged |
| `tests/test_examples.py` | gains `EXAMPLES_BY_MEMBER` and `test_every_package_has_a_gated_example`; `OFFLINE` and the two existing tests are untouched |

No production code moves. No package, example, or `pyproject.toml` changes.

## Data and interfaces

In `tests/test_package_boundaries.py`:

```python
#: A -> B means A depends on B. The five members that exist today; the four the
#: epic lists but does not carve (agents, chat, flow, telegram) join it when
#: their directories do.
DECLARED_GRAPH: dict[str, frozenset[str]] = {
    "scufris_core": frozenset(),
    "scufris_host": frozenset(),
    "scufris_hostd": frozenset({"scufris_core", "scufris_host"}),
    "scufris_hostctl": frozenset({"scufris_core", "scufris_host", "scufris_hostd"}),
    "scufris": frozenset({"scufris_core", "scufris_host", "scufris_hostd", "scufris_hostctl"}),
}

def _sibling_edges(roots: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    """member -> imported member -> the modules that import it."""

def _cycles(graph: dict[str, frozenset[str]]) -> list[list[str]]:
    """Every cycle reachable in `graph`, as node paths. Empty when acyclic."""

def _graph_problems(
    declared: dict[str, frozenset[str]],
    edges: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Every way the tree disagrees with `declared`, as messages."""
```

`_graph_problems` reports four arms, all of them, not the first:

1. a member on disk that `declared` does not mention (and vice versa),
2. a real edge `declared` does not allow, naming the importing modules,
3. a declared edge with no real import behind it (a stale declaration),
4. a cycle in `declared`.

In `tests/test_examples.py`:

```python
#: The example that PROVES each workspace member. Keyed by import root, so a new
#: member under packages/ cannot appear without an entry here. The mapping is
#: not the filename's: `host`'s example is host_report_fixture.py, and the root's
#: is host_agent.py, the only offline example that boots the composition root.
EXAMPLES_BY_MEMBER = {
    "scufris_core": "core_unit_of_work.py",
    "scufris_host": "host_report_fixture.py",
    "scufris_hostd": "hostd_socket_roundtrip.py",
    "scufris_hostctl": "hostctl_approval_flow.py",
    "scufris": "host_agent.py",
}
```

The member set comes from `test_package_boundaries._import_roots()` - imported,
not re-derived, so the two gates cannot disagree about what a member is.

## Sketches

Illustrative, not the patch.

```python
# tests/test_package_boundaries.py
def test_package_import_graph_matches_the_declared_graph() -> None:
    roots = _import_roots()
    assert _graph_problems(DECLARED_GRAPH, _sibling_edges(roots)) == []

def test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle() -> None:
    """The falsifier. Without it, a checker that returns [] is green forever."""
    edges = {"scufris_core": {"scufris_host": ["engine.py"]}, ...}
    assert [p for p in _graph_problems(DECLARED_GRAPH, edges) if "not allowed" in p]
    assert _cycles({"a": frozenset({"b"}), "b": frozenset({"a"})})
```

```python
# tests/test_examples.py
def test_every_package_has_a_gated_example() -> None:
    members = set(test_package_boundaries._import_roots())
    assert set(EXAMPLES_BY_MEMBER) == members          # no member unclaimed
    for member, name in sorted(EXAMPLES_BY_MEMBER.items()):
        assert name in OFFLINE                          # the gate runs it
        assert (EXAMPLES / name).is_file()
        assert member in _imports(EXAMPLES / name)      # the claim is not free
```

## Shape

```text
  packages/*/src/*  +  scufris/          examples/*.py + OFFLINE
            |                                     |
      _import_roots()  <-- one member list -->  EXAMPLES_BY_MEMBER
            |                                     |
     _sibling_edges()                     every member claims an
            |                             example that is gated,
     _graph_problems(DECLARED_GRAPH, .)   exists, and imports it
        /        |        \
   undeclared  stale   cycle
      edge     edge
```

## Consequences and open questions

- Cost: two literals to maintain. Every future carve edits `DECLARED_GRAPH` and
  `EXAMPLES_BY_MEMBER` or goes red. That is the gate working, not friction to
  design away, and it is the same bargain `CORE_MODULES` already strikes.
- Assumption recorded, not blocking: the graph is checked for EQUALITY, not
  containment. A declared edge whose last real import disappears fails the test
  until the line is deleted. Containment would let the declaration drift into
  fiction, which is the failure this task exists to close.
- Assumption recorded: the root `scufris` is a member of both gates, per
  TASK.md. Its example is `host_agent.py` - the only offline example that
  imports `scufris` - and the name is a poor fit for what it proves. Renaming it
  is out of scope; the mapping literal is where the mismatch is written down.
- The claim check (`member in imports of its example`) is one-directional: it
  stops a member claiming an unrelated file, but any example importing
  `scufris_core` would satisfy `core`'s claim. Naming exactly one example per
  member is what keeps that honest, not the assertion.
- `hostctl_approval_flow.py` imports four members. Under a derived
  "an example covers what it imports" rule it would cover the whole workspace on
  its own, which is why the mapping is declared rather than inferred.
- Open, and deferred with the epic: `agents`, `chat`, `flow`, `telegram` have
  declared edges in the epic but no directories. They are deliberately absent
  from both literals; adding them before the code exists would assert nothing.
