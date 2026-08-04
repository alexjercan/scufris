"""The carve's three claims, as checks instead of README paragraphs.

`core` is generic, every package keeps its hands off its siblings' internals,
and the dependencies run in the declared direction. Those are the properties the
whole workspace split is FOR, and all three are the kind that decay one
plausible-looking import at a time. They are checked here by reading the source
tree with `ast`, not by importing it: a check that imports cannot report on a
package whose dependencies are not installed yet, and every later carve task
adds a member before its wiring is finished.

`DECLARED_GRAPH` is the third claim, and the epic's dependency direction as a
literal rather than a README paragraph. It is checked for EQUALITY against the
sibling imports the tree really makes - a declared edge whose last import
disappears has to fail, or the declaration drifts into fiction - and for
acyclicity. `test_no_package_imports_a_sibling_private_module` polices HOW a
sibling is reached; this one polices WHICH siblings may be reached at all.

`test_core_is_domain_free` is an ALLOWLIST, deliberately. A pure property check
- "declares no table" - is satisfied trivially by everything `core` is already
planned to gain (`EventBus`, the generic `Supervisor`, `RunPhase`), so it would
wave through exactly the junk-drawer decay it is named for. Adding a module to
`core` is meant to cost a line here and a justification in the task record.

`test_the_domain_free_check_rejects_the_pre_move_tree` is the falsifier: the
same helper, pointed at `scufris/db`, where `Base` still sits beside eleven
row classes. Without it "core is domain-free" is a green light that would stay
green if the helper checked nothing at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"

#: The modules `scufris_core` is allowed to contain. `logsetup` is on the list
#: by a recorded decision: it is 87 generic lines imported by modules the carve
#: splits across four future packages, and no package can become a workspace
#: member while it imports a root module for its logging. `eventbus` is on it
#: for the same shape of reason: it is generic over its payload, it imports
#: nothing but the standard library, and its second consumer (`hostctl`) is in
#: another distribution that would otherwise have to import the app for a bus.
#: `supervisor` is the generic half of the run engine, split from the agent's
#: instantiation for the same reason: `hostctl` supervises applies and config
#: builds, and the event type is a parameter.
CORE_MODULES = frozenset({"", "engine", "base", "logsetup", "eventbus", "supervisor"})


def _modules(package_root: Path) -> dict[str, Path]:
    """Dotted module name -> file, for every module under `package_root`.

    The package's own `__init__.py` is named `""`, so a caller's allowlist can
    talk about the facade without a special case.
    """
    found: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


def _base_names(node: ast.ClassDef) -> set[str]:
    """The bases of `node`, by their last dotted segment."""
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _declares_tablename(node: ast.ClassDef) -> bool:
    """Does `node`'s body bind `__tablename__`?"""
    for statement in node.body:
        targets: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        if any(
            isinstance(target, ast.Name) and target.id == "__tablename__"
            for target in targets
        ):
            return True
    return False


#: Base names that make a class declarative. `DeclarativeBase` is SQLAlchemy's;
#: `Base` is named too because it is the workspace's single shared base BY
#: CONTRACT and it now lives in another distribution - a tree that subclasses it
#: without defining it (which is every tree but `core`) would otherwise look
#: like eleven ordinary classes.
DECLARATIVE_ROOTS = frozenset({"DeclarativeBase", "Base"})


def _declarative_classes(
    classes: dict[tuple[str, str], ast.ClassDef],
) -> set[tuple[str, str]]:
    """Every `(module, class)` in `classes` reaching a declarative root by bases.

    Transitive, so an intermediate mixin between a row and `Base` does not hide
    the row from the check. Keyed by module AND name because two modules may
    each define a class of the same name, and keying by the bare name alone
    would drop one of them out of the walk entirely. Bases are matched against
    the bare names, since an `ast` base node carries no module.
    """
    reached: set[tuple[str, str]] = set()
    while True:
        roots = DECLARATIVE_ROOTS | {name for _, name in reached}
        grown = {key for key, node in classes.items() if _base_names(node) & roots}
        if grown <= reached:
            return reached
        reached |= grown


def _domain_free(package_root: Path, allowed: frozenset[str]) -> list[str]:
    """Every way `package_root` fails the domain-free rule, as messages.

    Three arms: no module outside `allowed`, no module declaring a table, and no
    declarative class but `Base` itself. Returns an empty list when the tree is
    clean, and reports ALL failures rather than the first.
    """
    problems = []
    modules = _modules(package_root)
    classes: dict[tuple[str, str], ast.ClassDef] = {}
    for name, path in sorted(modules.items()):
        if name not in allowed:
            problems.append(
                f"{name or '__init__'}: not on the allowlist. Adding a module to "
                f"{package_root.name} is a deliberate act - justify it in the "
                "task record and add it to the allowlist this check was called "
                "with."
            )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            classes[(name, node.name)] = node
            if _declares_tablename(node):
                problems.append(
                    f"{name or '__init__'}: {node.name} declares __tablename__; "
                    "domain tables belong to the package that owns them"
                )
    for module, declarative in sorted(_declarative_classes(classes)):
        if declarative != "Base":
            problems.append(
                f"{module or '__init__'}: {declarative} is a declarative class; "
                "the only one here may be Base"
            )
    return problems


def _imported_modules(path: Path) -> set[str]:
    """Every ABSOLUTE module name `path` imports.

    Relative imports are skipped: they cannot name a sibling distribution.
    """
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


def _import_roots() -> dict[str, Path]:
    """Import root -> source directory, for every workspace member on disk.

    The root distribution `scufris` is one of them, and not a special case: it
    is a member in `uv.lock`, it is by far the largest consumer of the packages,
    and it is the tree the carve moves code OUT of - so it is the one most
    likely to keep an import pointing at where a module used to sit.
    """
    roots = {
        source.name: source
        for source in sorted(PACKAGES.glob("*/src/*"))
        if source.is_dir()
    }
    roots["scufris"] = REPO_ROOT / "scufris"
    return roots


#: A -> B means A depends on B. The five members that exist today; the four the
#: epic lists but does not carve (agents, chat, flow, telegram) join it when
#: their directories do. Checked for EQUALITY, not containment: a declared edge
#: whose last real import disappears fails until the line is deleted, because a
#: declaration allowed to drift into fiction is the failure this check exists to
#: close.
DECLARED_GRAPH: dict[str, frozenset[str]] = {
    "scufris_core": frozenset(),
    "scufris_host": frozenset(),
    "scufris_hostd": frozenset({"scufris_core", "scufris_host"}),
    "scufris_hostctl": frozenset({"scufris_core", "scufris_host", "scufris_hostd"}),
    "scufris": frozenset(
        {"scufris_core", "scufris_host", "scufris_hostd", "scufris_hostctl"}
    ),
}


def _sibling_edges(roots: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    """member -> imported member -> the modules that import it.

    Only the HEAD segment of an import matters here: which member is reached is
    the graph's question, and `test_no_package_imports_a_sibling_private_module`
    already owns whether the reach went through the facade or around it.
    """
    importers: dict[str, dict[str, set[str]]] = {name: {} for name in roots}
    for name, source in roots.items():
        for module, path in sorted(_modules(source).items()):
            for imported in sorted(_imported_modules(path)):
                head = imported.partition(".")[0]
                if head in roots and head != name:
                    importers[name].setdefault(head, set()).add(module or "__init__")
    return {
        name: {target: sorted(modules) for target, modules in sorted(reached.items())}
        for name, reached in importers.items()
    }


def _cycles(graph: dict[str, frozenset[str]]) -> list[list[str]]:
    """Every cycle reachable in `graph`, as node paths. Empty when acyclic.

    Deduplicated by the SET of nodes forming the cycle, so one representative
    path is reported per such set rather than every rotation of it. Not per
    strongly connected component: two cycles over overlapping nodes are two
    entries, which is what the falsifier's four-node graph asserts. The caller
    asks whether a cycle exists and which members are in it; enumerating each
    way round one would only pad the failure message.

    Every simple path is walked, with no memo of nodes already explored. A memo
    would make this linear in edges, but it drops cycles: mark a node done in
    one branch and a second cycle reaching it through another branch is never
    reported. Reporting every cycle is worth more here than an asymptotic win
    over a graph of five nodes that the epic grows to nine.
    """
    found: dict[frozenset[str], list[str]] = {}

    def walk(node: str, stack: list[str]) -> None:
        if node in stack:
            cycle = stack[stack.index(node) :]
            found.setdefault(frozenset(cycle), [*cycle, node])
            return
        for neighbour in sorted(graph.get(node, frozenset())):
            walk(neighbour, [*stack, node])

    for start in sorted(graph):
        walk(start, [])
    return [found[key] for key in sorted(found, key=sorted)]


def _graph_problems(
    declared: dict[str, frozenset[str]],
    edges: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Every way the tree disagrees with `declared`, as messages.

    Four arms, all of them rather than the first: a member set mismatch in
    either direction, a real edge `declared` does not allow, a declared edge
    with no real import behind it, and a cycle in `declared`.
    """
    problems = []
    for member in sorted(set(declared) - set(edges)):
        problems.append(
            f"{member}: in the declared graph but not a workspace member on "
            "disk; delete the entry or carve the package"
        )
    for member in sorted(set(edges) - set(declared)):
        problems.append(
            f"{member}: a workspace member on disk but missing from the "
            "declared graph; declaring what a new package may import is part "
            "of carving it"
        )
    for member in sorted(set(declared) & set(edges)):
        allowed, real = declared[member], edges[member]
        for target in sorted(set(real) - set(allowed)):
            importers = ", ".join(real[target])
            problems.append(
                f"{member} -> {target}: not allowed by the declared graph "
                f"(imported by {importers})"
            )
        for target in sorted(set(allowed) - set(real)):
            problems.append(
                f"{member} -> {target}: declared but nothing imports it; the "
                "graph is a claim about the tree, so a dead edge is a false one"
            )
    for cycle in _cycles(declared):
        problems.append(f"cycle in the declared graph: {' -> '.join(cycle)}")
    return problems


def test_the_graph_check_rejects_an_undeclared_edge_and_a_cycle() -> None:
    """The falsifier. Without it, a checker that returns [] is green forever.

    All four arms are driven THROUGH `_graph_problems` rather than by pinning
    its helpers: a helper that works proves nothing once the checker stops
    calling it, and deleting either the cycle loop or the dead-edge loop used to
    leave this file green.

    Hand-built inputs, not the tree: the real edges match `DECLARED_GRAPH`
    today, so every arm that matters here would be unreachable from disk.
    """
    edges: dict[str, dict[str, list[str]]] = {name: {} for name in DECLARED_GRAPH}
    edges["scufris_core"] = {"scufris_host": ["engine.py"]}
    problems = _graph_problems(DECLARED_GRAPH, edges)
    assert [p for p in problems if "not allowed" in p], problems
    assert [p for p in problems if "engine.py" in p], problems
    assert [p for p in problems if "declared but nothing imports it" in p], problems

    cyclic = _graph_problems(
        {"a": frozenset({"b"}), "b": frozenset({"a"})},
        {"a": {"b": ["x.py"]}, "b": {"a": ["y.py"]}},
    )
    assert [p for p in cyclic if p.startswith("cycle in the declared graph")], cyclic

    two_cycles = _graph_problems(
        {
            "b": frozenset({"c", "x"}),
            "c": frozenset({"m"}),
            "x": frozenset({"m"}),
            "m": frozenset({"b"}),
        },
        {},
    )
    reported = [p for p in two_cycles if p.startswith("cycle in the declared")]
    assert len(reported) == 2, reported

    extra = _graph_problems(DECLARED_GRAPH, {**edges, "scufris_agents": {}})
    assert [p for p in extra if p.startswith("scufris_agents: a workspace")], extra

    dropped = {name: reached for name, reached in edges.items() if name != "scufris"}
    missing = _graph_problems(DECLARED_GRAPH, dropped)
    assert [p for p in missing if p.startswith("scufris: in the declared")], missing


def test_package_import_graph_matches_the_declared_graph() -> None:
    """The declared dependency direction, checked against the real imports."""
    roots = _import_roots()
    assert roots, f"no workspace member found under {PACKAGES}/*/src/*"
    assert _graph_problems(DECLARED_GRAPH, _sibling_edges(roots)) == []


def test_core_is_domain_free() -> None:
    core = PACKAGES / "core" / "src" / "scufris_core"
    assert core.is_dir(), f"{core} does not exist"
    assert _domain_free(core, CORE_MODULES) == []


def test_the_domain_free_check_rejects_the_pre_move_tree() -> None:
    """The same helper, against a tree that is emphatically not domain-free.

    `allowed` is the pre-move tree's own module list, which neutralises the
    allowlist arm on purpose: what is being proven here is that the two DOMAIN
    arms bite, not that an unexpected filename is noticed.
    """
    pre_move = REPO_ROOT / "scufris" / "db"
    problems = _domain_free(pre_move, frozenset(_modules(pre_move)))
    assert [p for p in problems if "__tablename__" in p], problems
    assert [p for p in problems if "is a declarative class" in p], problems


def test_no_package_imports_a_sibling_private_module() -> None:
    """A member may import a sibling's distribution root, never its internals.

    `from scufris_core import Database` is the contract; `from
    scufris_core.engine import Database` reaches around it and pins a layout the
    owning package is free to change. With `core` and the root as the only two
    members the rule has a single pair to police; it earns a red run once a
    second package is carved out beside `core`.
    """
    roots = _import_roots()
    assert roots, f"no workspace member found under {PACKAGES}/*/src/*"
    problems = []
    for name, source in roots.items():
        siblings = set(roots) - {name}
        for module, path in sorted(_modules(source).items()):
            for imported in sorted(_imported_modules(path)):
                head, _, rest = imported.partition(".")
                if head in siblings and rest:
                    problems.append(
                        f"{name}.{module or '__init__'} imports {imported}; "
                        f"import {head} itself, not its internals"
                    )
    assert problems == []
