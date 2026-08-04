"""Run the examples that can run unattended, and fail when one rots.

An example is documentation that executes, which is only worth more than prose
if something notices when it stops executing. This is that something.

`OFFLINE` is an explicit OPT-IN list, not "every file in `examples/`". Most of
the fourteen scripts there cannot run in a test process: `auth_session.py` boots a
real uvicorn on a real port, `host_inspect.py` and `nixos_change.py` need a
NixOS machine, and `telegram_bot.py` wants a token. There is no marker
distinguishing them yet, and inventing one to justify a blanket glob would be a
worse lie than a short list. A script joins the list when it is genuinely
offline; until then it is unproven, and the list says which are which.

Each runs as a SUBPROCESS rather than by import, because that is how an operator
runs it: `__main__`, a fresh interpreter, and the exit code as the verdict.

`EXAMPLES_BY_MEMBER` is the module's second claim, and what makes `OFFLINE`
more than a hand-written list: every workspace member names an example that is
on `OFFLINE`, exists, and imports the member it is claimed to prove. The member
set is IMPORTED from `test_package_boundaries` rather than re-derived, so the
two gates cannot disagree about what a member is. The claim is DECLARED rather
than inferred from what each example imports, because
`hostctl_approval_flow.py` imports all four packages and would otherwise cover
the workspace on its own.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_package_boundaries import _import_roots, _imported_modules

import scufris_chat
from scufris_chat import Actor

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Examples that need no network, no host and no operator. Add one only after
#: running it in a clean checkout.
OFFLINE = (
    "chat_conversation.py",
    "core_unit_of_work.py",
    "host_agent.py",
    "host_report_fixture.py",
    "hostctl_approval_flow.py",
    "hostd_socket_roundtrip.py",
)


#: The example that PROVES each workspace member. Keyed by import root, so a new
#: member under packages/ cannot appear without an entry here. The mapping is not
#: the filename's: `host`'s example is host_report_fixture.py, and the root's is
#: host_agent.py, the only offline example that boots the composition root.
EXAMPLES_BY_MEMBER = {
    "scufris_chat": "chat_conversation.py",
    "scufris_core": "core_unit_of_work.py",
    "scufris_host": "host_report_fixture.py",
    "scufris_hostd": "hostd_socket_roundtrip.py",
    "scufris_hostctl": "hostctl_approval_flow.py",
    "scufris": "host_agent.py",
}


def _example_problems(
    members: set[str], claimed: dict[str, str], offline: tuple[str, ...] = OFFLINE
) -> list[str]:
    """Every way `claimed` fails to prove `members`, as messages.

    Four arms, all of them rather than the first: a member claiming no example,
    a claim for something that is not a member, an example nothing runs because
    it is off `offline`, and an example that is missing or does not import the
    member it is claimed to prove.

    `offline` is a parameter for the same reason `_domain_free` takes its
    allowlist as one: the falsifier drives the off-list arm with a list of its
    own, so putting a real script on `OFFLINE` later cannot turn it red for a
    reason unrelated to the arm it tests.
    """
    problems = []
    for member in sorted(members - set(claimed)):
        problems.append(
            f"{member}: a workspace member that claims no example; a package is "
            "done because its example runs, so claiming one is part of carving it"
        )
    for member in sorted(set(claimed) - members):
        problems.append(
            f"{member}: claims an example but is not a workspace member on "
            "disk; delete the entry or carve the package"
        )
    for member, name in sorted(claimed.items()):
        if member not in members:
            continue
        script = EXAMPLES / name
        if name not in offline:
            problems.append(f"{member}: {name} is not on OFFLINE, so nothing runs it")
        if not script.is_file():
            problems.append(f"{member}: {script} does not exist")
            continue
        reached = {imported.partition(".")[0] for imported in _imported_modules(script)}
        if member not in reached:
            problems.append(
                f"{member}: {name} does not import it, so it proves nothing"
            )
    return problems


def test_the_example_gate_rejects_an_unclaimed_member_and_a_rotted_claim() -> None:
    """The falsifier, for the same reason `_graph_problems` has one.

    Hand-built inputs, not the tree: the real claims cover the real members, so
    every arm here is unreachable from disk, and an arm no test drives is an arm
    that can be deleted without anything going red.
    """
    real = set(EXAMPLES_BY_MEMBER)
    unclaimed = _example_problems(real | {"scufris_agents"}, EXAMPLES_BY_MEMBER)
    assert [p for p in unclaimed if p.startswith("scufris_agents: a workspace")], (
        unclaimed
    )
    stale = _example_problems(real - {"scufris"}, EXAMPLES_BY_MEMBER)
    assert [p for p in stale if p.startswith("scufris: claims an example")], stale

    core = {"scufris_core"}
    off_list = _example_problems(core, {"scufris_core": "host_agent.py"}, ())
    assert [p for p in off_list if "not on OFFLINE" in p], off_list

    unrelated = {"scufris_core": "host_report_fixture.py"}
    rotted = _example_problems(core, unrelated, ("host_report_fixture.py",))
    assert [p for p in rotted if "does not import it" in p], rotted
    absent = _example_problems(core, {"scufris_core": "no_such.py"}, ("no_such.py",))
    assert [p for p in absent if "does not exist" in p], absent


def test_every_package_has_a_gated_example() -> None:
    """A package is done because its example runs, so every member needs one.

    `OFFLINE` alone cannot prove that: it is an opt-in tuple with no relation to
    the member list, so a package that ships no example - or ships one and never
    adds it - leaves the gate green. The member set is IMPORTED from
    `test_package_boundaries` rather than re-derived, so the two gates cannot
    disagree about what a member is.

    The claim check is one-directional by design: it stops a member claiming an
    unrelated file, but any example importing `scufris_core` would satisfy
    `core`'s claim. Naming exactly one example per member is what keeps that
    honest.
    """
    members = set(_import_roots())
    assert len(members) > 1, "no carved member found; only the root distribution"
    assert _example_problems(members, EXAMPLES_BY_MEMBER) == []


@pytest.mark.parametrize("name", OFFLINE)
def test_offline_example_runs(name: str) -> None:
    script = EXAMPLES / name
    assert script.is_file(), f"{script} does not exist"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_host_report_fixture_calls_every_renderer() -> None:
    """`host_report_fixture.py` covers all of `scufris_host.render`.

    Running the script only proves the renderers it happens to call still work.
    A renderer added to `render.py` and never called from anywhere would keep
    this suite green while being covered by nothing, so the example's claim to
    render EVERY report is checked as a claim rather than trusted.
    """
    source = (EXAMPLES / "host_report_fixture.py").read_text(encoding="utf-8")
    renderers = REPO_ROOT / "packages" / "host" / "src" / "scufris_host" / "render.py"
    names = {
        node.name
        for node in ast.parse(renderers.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("render_")
    }
    assert names, f"no renderers found in {renderers}"
    assert sorted(n for n in names if f"render.{n}(" not in source) == []


CHAT_DEMO = "chat_conversation.py"


def _called_names(script: Path) -> list[str]:
    """Every bare `name(...)` call in `script`, in source order, repeats kept.

    Repeats matter: how many times the demo calls `append_event` is how long the
    transcript it renders is, and a set would throw that away.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def _load_example(name: str) -> types.ModuleType:
    """Import an example for the constants it declares. `main` does not run.

    The gates above treat an example as a subprocess and an exit code. This one
    needs the demo's own tables - which actors it writes, which guides its tree
    draws - and re-typing them here would let the assertion and the demo drift
    apart in the one direction that keeps this green.
    """
    spec = importlib.util.spec_from_file_location(
        f"example_{name[:-3]}", EXAMPLES / name
    )
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_lines(demo: types.ModuleType, stdout: str, event_seq: int) -> list[str]:
    """Every rendered line carrying event `event_seq`, its guides still attached.

    A depth of zero is not one of them. The demo numbers its steps as well as
    its events, and a step header sits flush against the left margin while every
    event is drawn inside the tree - which is the same fact this asserts on.

    The guides and the depth rule come off `demo` for the reason `_load_example`
    gives: a copy here would drift from what the demo actually draws.
    """
    return [
        line
        for line in stdout.splitlines()
        if demo.depth(line) > 0
        and line.lstrip(demo.GUIDE_CHARACTERS).startswith(f"{event_seq}. ")
    ]


def _causation(script: Path) -> tuple[int, dict[int, int]]:
    """How many events the demo appends, and which event each one answers.

    The Nth `append_event` call is event N - the same rule `_called_names`
    counts by - and a `causation_id=<name>.id` keyword names the call whose
    result was bound to `<name>`, so the edges are read back out of the demo
    rather than typed here. A demo that rewires which event answers which is
    then asserted against its new shape instead of an assumed one.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    bound = {
        node.value: node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "append_event"
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    seq_of = {
        bound[call]: seq for seq, call in enumerate(calls, start=1) if call in bound
    }
    causes: dict[int, int] = {}
    for seq, call in enumerate(calls, start=1):
        for keyword in call.keywords:
            value = keyword.value
            if (
                keyword.arg == "causation_id"
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
            ):
                causes[seq] = seq_of[value.value.id]
    return len(calls), causes


def test_chat_conversation_calls_every_exported_function() -> None:
    """`chat_conversation.py` calls every function `scufris_chat` exports.

    The example is Lane 1's deliverable, and the claim it makes is that the
    WHOLE package runs: a function exported and never called by it is a corner
    of the lane the demo does not prove. Running the script cannot notice - it
    exits 0 on the subset it happens to touch - so the claim is checked as a
    claim, the way `test_host_report_fixture_calls_every_renderer` checks its
    example's.

    `__all__` rather than `dir()`: the public surface is what the demo owes a
    call to, and a private helper is not part of it.
    """
    exported = {
        name
        for name in scufris_chat.__all__
        if isinstance(getattr(scufris_chat, name), types.FunctionType)
    }
    assert exported, "scufris_chat exports no functions"
    called = set(_called_names(EXAMPLES / CHAT_DEMO))
    assert sorted(exported - called) == []


def test_chat_conversation_renders_an_attributed_causation_tree() -> None:
    """The demo's OUTPUT is an ordered, attributed transcript with its causation.

    `test_offline_example_runs` judges the demo by its exit code, so everything
    it prints is unchecked by that gate: a transcript that lost its attribution,
    its order or its edges would still exit 0. This reads the output an operator
    reads.

    Every claim here is structural rather than a literal expected string - the
    events come from the demo's own `append_event` calls, the actor labels from
    the `Actor` constants it declares, the edges from its `TREE_GUIDES` - so
    editing the demo's wording does not turn this red and dropping half of what
    it renders does.
    """
    demo = _load_example(CHAT_DEMO)
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / CHAT_DEMO)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{CHAT_DEMO} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    stdout = result.stdout

    appended, causes = _causation(EXAMPLES / CHAT_DEMO)
    assert appended >= 2, "a transcript of one event has no causation to draw"
    assert causes, "the demo appends no event that answers another"
    for event_seq in range(1, appended + 1):
        assert len(_tree_lines(demo, stdout, event_seq)) == 2, (
            f"event {event_seq} should be rendered twice by the SAME renderer, "
            f"before the backend switch and after it\n{stdout}"
        )

    # Inside a rendered line rather than anywhere in the output: the demo names
    # every actor in its step 3 and step 6 prose as well, so `in stdout` would
    # pass on a tree that carries no attribution at all.
    rendered = [
        line
        for event_seq in range(1, appended + 1)
        for line in _tree_lines(demo, stdout, event_seq)
    ]
    attributed = {
        value.render() for value in vars(demo).values() if isinstance(value, Actor)
    }
    assert len(attributed) >= 2, "one actor attributes nothing"
    for label in sorted(attributed):
        assert any(label in line for line in rendered), (
            f"no rendered event names {label}\n{stdout}"
        )

    for guide in demo.TREE_GUIDES:
        assert guide in stdout, f"the transcript draws no {guide!r} edge\n{stdout}"

    for event_seq, cause_seq in sorted(causes.items()):
        answer = _tree_lines(demo, stdout, event_seq)[0]
        asked = _tree_lines(demo, stdout, cause_seq)[0]
        assert stdout.index(answer) > stdout.index(asked), (
            f"event {event_seq} is drawn before event {cause_seq}, "
            f"which it answers\n{stdout}"
        )
        assert demo.parent_line(stdout.splitlines(), answer) == asked, (
            f"event {event_seq} answers event {cause_seq} and should be drawn "
            f"UNDER it, not beside it or under something else\n{stdout}"
        )
