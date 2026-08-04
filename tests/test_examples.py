"""Run the examples that can run unattended, and fail when one rots.

An example is documentation that executes, which is only worth more than prose
if something notices when it stops executing. This is that something.

`OFFLINE` is an explicit OPT-IN list, not "every file in `examples/`". Most of
the twelve scripts there cannot run in a test process: `auth_session.py` boots a
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
import subprocess
import sys
from pathlib import Path

import pytest
from test_package_boundaries import _import_roots, _imported_modules

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Examples that need no network, no host and no operator. Add one only after
#: running it in a clean checkout.
OFFLINE = (
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
    assert members, "no workspace member found"
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
