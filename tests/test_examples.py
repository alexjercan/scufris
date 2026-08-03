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
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Examples that need no network, no host and no operator. Add one only after
#: running it in a clean checkout.
OFFLINE = ("core_unit_of_work.py", "host_report_fixture.py")


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
