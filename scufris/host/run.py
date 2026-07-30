"""The one door every host inspection goes through to reach a CLI.

Nothing in ``scufris.host`` calls ``subprocess`` directly. Every command runs
through a ``Runner`` and comes back as a ``CommandResult`` whose ``outcome``
names WHY it did not work - missing binary, denied, timed out, non-zero exit -
so "degrades explicitly" is a property of this layer rather than a habit each
inspection has to remember.

The runner is injectable, the way ``metrics.GpuRunner`` is: tests replay real
captured output through ``FakeRunner`` instead of patching subprocess internals,
and every failure mode is reachable through the same door as the success path.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Nix's new CLI (`nix path-info`, `nix store diff-closures`, `nix build`) sits
# behind experimental features, and whether they are enabled is the OPERATOR's
# nix.conf rather than anything this code controls. Measured in the hostd VM
# test: with a default nix.conf, `nix path-info` fails outright with
# "experimental Nix feature 'nix-command' is disabled" - so a preview, a
# validation or a build that assumed otherwise degrades on a machine whose only
# fault is not having opted in. Every invocation therefore carries the features
# explicitly, exactly as `nixos-rebuild` does (FLAKE_FLAGS in
# nixos_rebuild/nix.py). The old CLI (`nix-env`, `nix-store`) needs none of this.
NIX_FEATURES = ["--extra-experimental-features", "nix-command flakes"]


def nix_cli(*args: str) -> list[str]:
    """`nix <args>`, with the experimental features it needs made explicit."""
    return ["nix", *NIX_FEATURES, *args]


# Default wall-clock bound for an inspection command. Individual callers raise it
# where the work is genuinely long (a store walk), never lower it silently.
DEFAULT_TIMEOUT = 10.0

# Stderr from a command that failed because we are not root. Recognising this
# separately from a generic non-zero exit is what lets a report say "needs
# privilege" instead of the useless "exit 4".
_DENIED = re.compile(
    r"permission denied|operation not permitted|must be root|are you root|"
    r"access denied|not authorized|insufficient privileges",
    re.IGNORECASE,
)


class Outcome(StrEnum):
    """Why a command produced what it produced.

    ``OK`` is the only outcome carrying trustworthy stdout. The rest each map to
    a different sentence in a report, which is the entire point of splitting
    them: an operator reading "journalctl is not installed" acts differently from
    one reading "journalctl needs privilege here".
    """

    OK = "ok"
    MISSING = "missing"  # the binary is not on PATH
    DENIED = "denied"  # it ran and refused for lack of privilege
    TIMEOUT = "timeout"  # it exceeded its wall-clock bound
    FAILED = "failed"  # any other non-zero exit


class CommandResult(BaseModel):
    """One command's outcome, stdout and the reason it failed if it did."""

    argv: list[str]
    outcome: Outcome
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    @property
    def name(self) -> str:
        return self.argv[0] if self.argv else "(no command)"

    def reason(self) -> str:
        """A sentence an operator can act on, empty when the command succeeded.

        The stderr tail is included for the failure cases where it carries the
        actual diagnosis; it is bounded because a failing command can be chatty.
        """
        if self.ok:
            return ""
        detail = " ".join(self.stderr.split())[:200]
        if self.outcome is Outcome.MISSING:
            return f"{self.name} is not installed on this host"
        if self.outcome is Outcome.DENIED:
            return f"{self.name} needs privilege here: {detail}"
        if self.outcome is Outcome.TIMEOUT:
            return f"{self.name} did not finish within its time budget"
        return (
            f"{self.name} failed (exit {self.returncode}): {detail}"
            if detail
            else (f"{self.name} failed (exit {self.returncode})")
        )


@runtime_checkable
class Runner(Protocol):
    """The seam every inspection depends on. Tests provide a fake."""

    def __call__(
        self, argv: list[str], *, timeout: float = DEFAULT_TIMEOUT
    ) -> CommandResult:
        """Run ``argv`` and return its outcome. Never raises."""
        ...


def run_command(argv: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> CommandResult:
    """Run a curated command with a fixed argument list and no shell.

    ``shell=False`` with an explicit argv, the executable resolved on PATH, and
    every failure classified rather than raised - the caller decides how to say
    "unavailable", it never has to handle an exception.
    """
    exe = shutil.which(argv[0])
    if exe is None:
        logger.info("host: %s not on PATH", argv[0])
        return CommandResult(argv=argv, outcome=Outcome.MISSING)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.info("host: %s timed out after %ss", argv[0], timeout)
        return CommandResult(
            argv=argv,
            outcome=Outcome.TIMEOUT,
            duration_seconds=time.monotonic() - started,
        )
    except OSError as exc:
        # An executable we resolved but cannot exec (bad interpreter, ENOMEM).
        logger.info("host: %s could not run: %s", argv[0], exc)
        return CommandResult(argv=argv, outcome=Outcome.FAILED, stderr=str(exc))
    elapsed = time.monotonic() - started
    if proc.returncode == 0:
        return CommandResult(
            argv=argv,
            outcome=Outcome.OK,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=0,
            duration_seconds=elapsed,
        )
    outcome = Outcome.DENIED if _DENIED.search(proc.stderr) else Outcome.FAILED
    logger.info("host: %s -> exit=%d (%s)", argv[0], proc.returncode, outcome)
    return CommandResult(
        argv=argv,
        outcome=outcome,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        duration_seconds=elapsed,
    )


class FakeRunner(BaseModel):
    """A ``Runner`` that replays canned results, keyed by a command prefix.

    Lives in the package rather than the tests so the example script and any
    future harness can drive an inspection without touching the real host. Keys
    are matched as a prefix of the argv joined by spaces, longest first, so
    ``"systemctl show"`` can override a broader ``"systemctl"`` entry.
    """

    results: dict[str, CommandResult] = Field(default_factory=dict)
    default: CommandResult | None = None
    calls: list[list[str]] = Field(default_factory=list)

    def __call__(
        self, argv: list[str], *, timeout: float = DEFAULT_TIMEOUT
    ) -> CommandResult:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for key in sorted(self.results, key=len, reverse=True):
            if joined.startswith(key):
                return self.results[key].model_copy(update={"argv": list(argv)})
        if self.default is not None:
            return self.default.model_copy(update={"argv": list(argv)})
        return CommandResult(argv=list(argv), outcome=Outcome.MISSING)


def ok_result(stdout: str) -> CommandResult:
    """A successful ``CommandResult`` carrying ``stdout`` (test/example helper)."""
    return CommandResult(argv=[], outcome=Outcome.OK, stdout=stdout, returncode=0)
