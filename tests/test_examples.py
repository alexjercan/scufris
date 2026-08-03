"""Run the examples that can run unattended, and fail when one rots.

An example is documentation that executes, which is only worth more than prose
if something notices when it stops executing. This is that something.

`OFFLINE` is an explicit OPT-IN list, not "every file in `examples/`". Most of
the eleven scripts there cannot run in a test process: `auth_session.py` boots a
real uvicorn on a real port, `host_inspect.py` and `nixos_change.py` need a
NixOS machine, and `telegram_bot.py` wants a token. There is no marker
distinguishing them yet, and inventing one to justify a blanket glob would be a
worse lie than a short list. A script joins the list when it is genuinely
offline; until then it is unproven, and the list says which are which.

Each runs as a SUBPROCESS rather than by import, because that is how an operator
runs it: `__main__`, a fresh interpreter, and the exit code as the verdict.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Examples that need no network, no host and no operator. Add one only after
#: running it in a clean checkout.
OFFLINE = ("core_unit_of_work.py",)


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
